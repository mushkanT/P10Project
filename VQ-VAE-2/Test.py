import vq_vae_model as vq_vae
import tensorflow as tf



m = vq_vae.VQVAEModel(256)

print(m.model.summary())
tf.keras.utils.plot_model(m.model, to_file='vqvae.png', show_shapes=True, expand_nested=True, show_layer_names=True)
#out = encoder(inp)
#decoder = m.decode_2_layer(out[0]['quantize'], out[1]['quantize'])
#print(decoder.summary())

#img = decoder([out[0]['quantize'],out[1]['quantize']])
