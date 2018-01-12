user_input = Input(shape=(1,), dtype=np.int32, name="user_input")
item_input = Input(shape=(1,), dtype=np.int32, name="item_input")

layers = [20, 10]
mlp_dim = layers[0] // 2
MLP_Embedding_User = Embedding(input_dim=num_users, 
                               output_dim=mlp_dim,
                               name="mlp_user_embedding",
                               embeddings_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                               embeddings_regularizer = l2(0.0), input_length=1) 
# MF_Embedding_User = np.random.rand(num_users, latent_dim) * 0.01
MLP_Embedding_Item = Embedding(input_dim=num_items,
                               output_dim=mlp_dim,
                               name="mlp_item_embedding",
                               embeddings_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                               embeddings_regularizer = l2(0.0), input_length=1) 
# MF_Embedding_Item = np.random.rand(latent_dim, num_items) * 0.01

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

prediction = Dense(units=1, 
                   activation="sigmoid", # softmax, sigmoid
                   use_bias=True,
                   bias_initializer=initializers.zeros(),
                   kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                   name = "prediction")(mlp_vector)

mlp_model = Model(inputs=[user_input, item_input], outputs=prediction)

mlp_model.compile(loss="binary_crossentropy", 
                  optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=dcy), # , decay=dcy
                  metrics=["accuracy"])
early_stopping = EarlyStopping(monitor="loss", patience=2, mode="min", verbose=1)

users, items, labels = shuffle(users, items, labels, random_state=0)

mlp_model_fit = mlp_model.fit([np.array(users), np.array(items)], labels, batch_size=bs, epochs=epc, verbose=1, shuffle=True, callbacks=[early_stopping])

mlp_model.save("D:/my_project/Python_Project/iTravel/itravel_recommend/h5/mlp_model.h5", overwrite=True, include_optimizer=True)
#mlp_model.save_weights("D:/my_project/Python_Project/iTravel/itravel_recommend/h5/mlp_model_weights.h5", overwrite=True)
