user_input = Input(shape=(1,), dtype=np.int32, name="user_input")
item_input = Input(shape=(1,), dtype=np.int32, name="item_input")

gmf_dim = 8; layers = [20, 10]; mlp_dim = layers[0] // 2
# Embedding layer
GMF_Embedding_User = Embedding(input_dim=num_users, 
                               output_dim=gmf_dim,
                               name="gmf_user_embedding",
                               embeddings_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                               embeddings_regularizer = l2(0.0), input_length=1) 
GMF_Embedding_Item = Embedding(input_dim=num_items,
                               output_dim=gmf_dim,
                               name="gmf_item_embedding",
                               embeddings_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                               embeddings_regularizer = l2(0.0), input_length=1) 

MLP_Embedding_User = Embedding(input_dim=num_users, 
                               output_dim=mlp_dim,
                               name="mlp_user_embedding",
                               embeddings_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                               embeddings_regularizer = l2(0.0), input_length=1)
MLP_Embedding_Item = Embedding(input_dim=num_items,
                               output_dim=mlp_dim,
                               name="mlp_item_embedding",
                               embeddings_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                               embeddings_regularizer = l2(0.0), input_length=1) 

# MF part
gmf_user_latent = Flatten()(GMF_Embedding_User(user_input))
gmf_item_latent = Flatten()(GMF_Embedding_Item(item_input))
gmf_vector = Multiply()([gmf_user_latent, gmf_item_latent])

# MLP part 
mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
mlp_vector = concatenate([mlp_user_latent, mlp_item_latent], axis=1)

mlp_vector = Dense(units=layers[0], 
                   activation="relu", # softmax, sigmoid, relu
                   use_bias=True,
                   bias_initializer=initializers.zeros(),
                   kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                   name = "layer_1")(mlp_vector)

mlp_vector = Dense(units=layers[1], 
                   activation="relu", # softmax, sigmoid, relu
                   use_bias=True,
                   bias_initializer=initializers.zeros(),
                   kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                   name = "layer_2")(mlp_vector)

predict_vector = concatenate([gmf_vector, mlp_vector], axis=1)
prediction = Dense(units=1, 
                   activation="sigmoid", # softmax, sigmoid
                   use_bias=True,
                   bias_initializer=initializers.zeros(),
                   kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                   name = "prediction")(predict_vector)

neumf_model = Model(inputs=[user_input, item_input], outputs=prediction)

# load model
gmf_model = load_model("D:/my_project/Python_Project/iTravel/itravel_recommend/h5/gmf_model.h5", compile=True)
mlp_model = load_model("D:/my_project/Python_Project/iTravel/itravel_recommend/h5/mlp_model.h5", compile=True)

# GMF embeddings weights
gmf_user_embedding = gmf_model.get_layer("gmf_user_embedding").get_weights()
gmf_item_embedding = gmf_model.get_layer("gmf_item_embedding").get_weights()
neumf_model.get_layer("gmf_user_embedding").set_weights(gmf_user_embedding)
neumf_model.get_layer("gmf_item_embedding").set_weights(gmf_item_embedding)
# MLP embeddings weights
mlp_user_embedding = mlp_model.get_layer("mlp_user_embedding").get_weights()
mlp_item_embedding = mlp_model.get_layer("mlp_item_embedding").get_weights()
neumf_model.get_layer("mlp_user_embedding").set_weights(mlp_user_embedding)
neumf_model.get_layer("mlp_item_embedding").set_weights(mlp_item_embedding)
# layers weights
layer_1 = mlp_model.get_layer("layer_1").get_weights()
layer_2 = mlp_model.get_layer("layer_2").get_weights()
neumf_model.get_layer("layer_1").set_weights(layer_1)
neumf_model.get_layer("layer_2").set_weights(layer_2)
# Prediction weights
gmf_w, gmf_b = gmf_model.get_layer("prediction").get_weights()
mlp_w, mlp_b = mlp_model.get_layer("prediction").get_weights()

alpha = 0.5
new_w = np.concatenate((alpha*gmf_w, (1-alpha)*mlp_w), axis=0)
new_b = alpha*gmf_b + (1-alpha)*mlp_b
neumf_model.get_layer("prediction").set_weights([new_w, new_b])

bs = 1024; epc = 100; lr = 0.01; dcy = 0.01
# binary_crossentropy
neumf_model.compile(loss="binary_crossentropy", 
                    optimizer=SGD(lr=0.01, momentum=0.0, decay=dcy), # , decay=dcy
                    metrics=["accuracy"])
early_stopping = EarlyStopping(monitor="loss", patience=2, mode="min", verbose=1)

users, items, labels = shuffle(users, items, labels, random_state=0)

neumf_model_fit = neumf_model.fit([np.array(users), np.array(items)], labels, batch_size=bs, epochs=epc, verbose=1, shuffle=True, callbacks=[early_stopping])
