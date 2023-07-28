import tensorflow as tf
import numpy as np
from urllib.request import urlopen
model=tf.keras.models.load_model('./linet5')

formats = list(map(bytes, ['BMP', 'PNG', 'RAW'], ['utf-8']*3)) + [b'\xff\xd8']
decoders = {
    b'BMP': tf.io.decode_bmp,
    b'PNG': tf.io.decode_png,
    b'RAW': tf.io.decode_raw,
    b'\xff\xd8': tf.io.decode_jpeg
}

def process_img(url):
    
        img=urlopen(url)
        img=img.read()
        for format in formats:
            if format in img:
                print("Format Found")
                img = decoders[format](img, channels = 1)                
                print("Image decoded")
                image=tf.io.encode_jpeg(img,format='grayscale')
                print("Image encoded")
                tf.io.write_file('./static/image.jpeg',image)
                img = tf.image.resize(img, [32, 32])
                img = img[tf.newaxis, :]
                res=model.predict(img)
                return np.argmax(res)

if __name__=='__main__':
    link="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRKdhZOx43HGlTAokyDoGh3JlAU8m0Dlk5euLdfE2upw&s"
    print(process_img(link))
    # print(bytes('JFIF','utf-8') in process_img(link))
    
    