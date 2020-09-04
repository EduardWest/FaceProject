# Обнаружение лица на массиве данных mixed_data
from os import listdir #загружаем библиотеку для просмотра папок
from os.path import isdir
from PIL import Image # конвертация изображения в массив
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN # Трехкаскадная сверточная нейронная сеть

# извлечение массива лица из фотографии
def extract_face(filename, required_size=(160, 160)):
    # загрузка изображения
    image = Image.open(filename)
    # конвертация в RGB
    image = image.convert('RGB')
    # преобразование в массив
    pixels = asarray(image)
    # создание детектора, используя готовую модель каскадной сверточной сети
    detector = MTCNN()
    # модель обнаружения лица
    results = detector.detect_faces(pixels)
    # извлекаем данные для построения прямоугольника
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # извлечение массива лица
    face = pixels[y1:y2, x1:x2]
    # изменение массива лица 160 * 160
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# Загрузка и извлечение лиц в необходимую папку
def load_faces(directory):
    faces = list() 
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces

# загрузить набор данных, который содержит один подкаталог для каждого класса, который в свою очередь содержит фотографии 
def load_dataset(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        # пропуск любых файлов, кроме фотографий
        if not isdir(path):
            continue
        # Загружаем каждое лицо в подпапки
        faces = load_faces(path)
        # создание меток
        labels = [subdir for _ in range(len(faces))]
        print('Загружено {} примеров для: {}'.format(len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# Загрузка тренировочного набора данных
trainX, trainy = load_dataset('mixed_data/train/')
print(trainX.shape, trainy.shape)
# Загрузка тестового набора данных
testX, testy = load_dataset('mixed_data/val/')
# Сохраняем в  архивированном массиве
savez_compressed('mixed_data.npz', trainX, trainy, testX, testy)

# Функция для проверки работы сети
def check_face(filename):
    image = cv2.imread(filename)
    results = detector.detect_faces(image)
    bounding_box = results[0]['box']
    keypoints = results[0]['keypoints']
    cv2.rectangle(image,(bounding_box[0], bounding_box[1]),
              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0,155,255), 2)
    cv2.circle(image,(keypoints['left_eye']), 1, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 1, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 1, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 1, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 1, (0,155,255), 2)
    cv2.namedWindow("image")
    cv2.imshow("image",image)
    cv2.waitKey(0)

# рассчитаем 128-мерный вектор-признак для каждого лица, используя уже обученную сеть facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from keras.models import load_model

# извлечение вектора для одного лица
def get_embedding(model, face_pixels):
    # преобразуем значения в массиве пикселя под 'float32'
    face_pixels = face_pixels.astype('float32')
    # стандартизация каждого пмкселя
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # добавляем фиктивную ось
    samples = expand_dims(face_pixels, axis = 0)
    # создание вектора-признака
    yhat = model.predict(samples)
    return yhat[0]

# Загружаем архив
data = load('mixed_data.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
# Инициализация модели
model = load_model('facenet_keras.h5')
print('Loaded Model')
# извлекаем вектор для каждой фотографии тренировочного набора
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
# извлекаем вектор для каждой фотографии тестируемого набора
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)
# сохраняем в массив
savez_compressed('mixed_data.npz', newTrainX, trainy, newTestX, testy)
# Проверка работы сети facenet
def makr():
pos = 0
true_accept = 0
d = 0.01
for i in trainy:
    if trainy[i] != trainy[i + 1]:
        pos += 1
        continue
    sum = 0
    if pos == 146:
        break
    for k, l in zip(trainX[pos], trainX[pos + 1]):
        sum += (k - l) ** 2
    if sum  <= d:
        true_accept += 1
    pos += 1
        
print('Validation rate:{}'.format(true_accept / len(trainy)))

# Создание классификатора SVM для mixed_data
from sklearn.metrics import accuracy_score
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
# Загрузка набора данных
data = load('mixed_data.npz')
testX_faces = data['arr_2']
data = load('mixed_data-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# нормализация входящего вектора
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# декодирование меток
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# обучение модели
model = SVC(kernel = 'linear', probability=True)
model.fit(trainX, trainy)
# прогноз
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# Построение ROC-кривой
fpr["micro"], tpr["micro"], _ = roc_curve(testy.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color = 'darkorange',
         lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc = "lower right")
plt.show()

# Тестирование случайного человека
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# Прогноз
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# Извлечение имени
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: {}'.format(random_face_name[0]))
# Построение графика
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()

