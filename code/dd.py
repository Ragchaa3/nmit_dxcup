import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import Input
from tensorflow.keras import backend as K

# 1. Өгөгдлийг CSV файлаас унших
data = pd.read_csv('cough_X_test.csv')

# 2. Текстийн багануудыг шүүж авах (зөвхөн тоон өгөгдлийг ашиглах)
# Текстийн багануудыг хасах
data = data.select_dtypes(include=[np.number])  # зөвхөн тоон өгөгдлийг үлдээх

# 3. Өгөгдлийг -1 ба 1 хооронд масштабжуулах
scaler = MinMaxScaler(feature_range=(-1, 1))
real_data = scaler.fit_transform(data.to_numpy())  # Өгөгдлийг масштабжуулах

# Use 'learning_rate' instead of 'lr' for the Adam optimizer
optimizer = Adam(learning_rate=0.0002, beta_1=0.5)  # Тохируулж болох гиперпараметрууд

# 4. Генераторын загвар
def build_generator(input_dim, output_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # Input layer
    model.add(Dense(128, activation='relu'))  # input_dim оруулна
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='tanh'))  # Гаралтын хэсэг
    model.compile(loss='binary_crossentropy', optimizer=optimizer)  # Use explicitly defined optimizer
    return model

# 5. Дискриминаторын загвар
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # Input layer
    model.add(Dense(128, activation='relu'))  # input_dim оруулна
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Хуурамч/Бодит ангилал
    model.compile(loss='binary_crossentropy', optimizer=optimizer)  # Use explicitly defined optimizer
    return model

# 6. GAN загвар (Генератор болон Дискриминаторыг нэгтгэх)
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Генераторын сургалтанд дискриминатор оролцохгүй
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)  # Use explicitly defined optimizer
    return model

# 7. GAN сургалт хийх функц
def train_gan(generator, discriminator, gan, epochs, batch_size):
    for epoch in range(epochs):
        # Гадагшлах санамж авах
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))  # Санамж үүсгэх
        fake_data = generator.predict(noise)

        # Реал өгөгдөл
        real_batch = real_data[np.random.randint(0, real_data.shape[0], batch_size)]

        # Үнэн зөв болон хуурамч хаягууд
        valid_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Дискриминаторын алдагдал (реал болон fake өгөгдлөөр)
        d_loss_real = discriminator.train_on_batch(real_batch, valid_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Генераторын алдагдал
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Алдагдлыг хэвлэх
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss:.4f} - G Loss: {g_loss:}")

        # 2000 давталтын дараа генератор болон дискриминаторын сургалтын хурдыг өөрчлөх
        if epoch % 2000 == 0:
            print("Adjusting learning rate...")
            generator.optimizer.learning_rate.assign(0.0001)
            discriminator.optimizer.learning_rate.assign(0.0001)

# 8. Сургалтын параметрүүд
epochs = 1500  # Сургалтын давталтын тоо
batch_size = 35  # Багцын хэмжээ

# Өгөгдөл дээрх хэмжээ
input_dim = real_data.shape[1]  # Өгөгдлийн хэмжээнээс input_dim авах
output_dim = real_data.shape[1]  # Гаралтын хэмжээ

# Генератор, Дискриминатор болон GAN загваруудыг үүсгэх
generator = build_generator(input_dim, output_dim)
discriminator = build_discriminator(input_dim)
gan = build_gan(generator, discriminator)


# GAN сургалтыг эхлүүлэх
train_gan(generator, discriminator, gan, epochs, batch_size)

# 9. Үүсгэгдсэн өгөгдлийг харах функц
def generate_and_inspect_data(generator, num_samples, scaler):
    # Санамсаргүй тоонуудыг үүсгэх
    noise = np.random.normal(0, 1, (num_samples, generator.input_shape[1]))
    # Генератороор дамжуулан шинэ өгөгдлийг үүсгэх
    generated_data = generator.predict(noise)
    # Масштабжуулсан өгөгдлийг анхны утгад шилжүүлэх
    generated_data_original_scale = scaler.inverse_transform(generated_data)
    return generated_data_original_scale

# Үүсгэгдэх өгөгдлийн тоо
num_samples = 250
generated_data = generate_and_inspect_data(generator, num_samples, scaler)

# 10. Үүсгэгдсэн өгөгдлийг консолоор харах
print("Үүсгэгдсэн өгөгдлийн урьдчилсан харагдац:")
print(pd.DataFrame(generated_data, columns=data.columns))

# 11. Үүсгэгдсэн өгөгдлийг .csv файлд хадгалах
output_file = 'gan_generated_cough_X_test.csv'
pd.DataFrame(generated_data, columns=data.columns).to_csv(output_file, index=False)
print(f"Үүсгэгдсэн өгөгдлийг {output_file} файлд хадгаллаа.")