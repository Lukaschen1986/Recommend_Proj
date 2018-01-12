user_input = Input(shape=(1,), dtype=np.int32, name="user_input")
item_input = Input(shape=(1,), dtype=np.int32, name="item_input")

gmf_dim = 8
GMF_Embedding_User = Embedding(input_dim=num_users, 
                               output_dim=gmf_dim,
                               name="gmf_user_embedding",
                               embeddings_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                               embeddings_regularizer = l2(0.0), input_length=1) 
# MF_Embedding_User = np.random.rand(num_users, latent_dim) * 0.01
GMF_Embedding_Item = Embedding(input_dim=num_items,
                               output_dim=gmf_dim,
                               name="gmf_item_embedding",
                               embeddings_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                               embeddings_regularizer = l2(0.0), input_length=1) 
# MF_Embedding_Item = np.random.rand(latent_dim, num_items) * 0.01

gmf_user_latent = Flatten()(GMF_Embedding_User(user_input))
gmf_item_latent = Flatten()(GMF_Embedding_Item(item_input))
gmf_vector = Multiply()([gmf_user_latent, gmf_item_latent])

prediction = Dense(units=1, 
                   activation="sigmoid", # softmax, sigmoid
                   use_bias=True,
                   bias_initializer=initializers.zeros(),
                   kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                   name = "prediction")(gmf_vector)

gmf_model = Model(inputs=[user_input, item_input], outputs=prediction)
gmf_model.summary()
#plot_model(mf_model, to_file="GMF.png", show_shapes=True, show_layer_names=True)
bs = 1024; epc = 100; lr = 0.1; dcy = 0.01

# binary_crossentropy
gmf_model.compile(loss="binary_crossentropy", 
                 optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=dcy), # , decay=dcy
                 metrics=["accuracy"])
early_stopping = EarlyStopping(monitor="loss", patience=2, mode="min", verbose=1)

users, items, labels = shuffle(users, items, labels, random_state=0)

gmf_model_fit = gmf_model.fit([np.array(users), np.array(items)], labels, batch_size=bs, epochs=epc, verbose=1, shuffle=True, callbacks=[early_stopping])

gmf_model.save("D:/my_project/Python_Project/iTravel/itravel_recommend/h5/gmf_model.h5", overwrite=True, include_optimizer=True)
