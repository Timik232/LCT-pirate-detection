# LCT-pirate-detection
Хакатон "Лидеры Цифровой Трансформации" \
## Сервис проверки видеофайлов на нарушение авторских прав
## Команда
- Легоньков Роман
- Шерри Георгий 
- Деев Леонид
- Парфёнов Егор
- Комолов Тимур
## Запуск приложения

## Описание решения
Решение задачи по обнаружению пиратства в видеофайлах представляет собой сервис, который позволяет проверить видеофайл
на наличие лицензионного контента. На клиентской стороне пользователь загружает видеофайл, который отправляется на сервер.
На сервере видеофайл сохраняется в отдельную базу данных и отправляется в сервис с машинным обучением.

### Сервис с машинным обучением
В сервисе построена векторная база данных лицензионных видеофайлов. Каждое видео представлено в виде двух векторов: 
вектора видео и вектора аудио.
Вектор видео получен с помощью модели Vision Transformer (ViT), а вектор аудио с помощью модели PANNS 
(PANNS: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition).
Эмбеддинги сохраняются в базе данных LanceDB.


#### Процесс работы сервиса
После того, как видеофайл передаётся в сервис, из него извлекаются вектора видео и аудио по каждой секунде видео. 
Далее, вычисляется косинусное расстояние между векторами видео и аудио загруженного видеофайла и векторами 
лицензионных видеофайлов и осуществляется поиск по векторной базе данных, находится наиболее похожее видео.
Затем осуществляется сравнение по кадрам в найденном видео и в загруженном. Между массивами эмбеддингов лицензионного и 
пиратского видео вычисляется косинусное расстояние и записи о косинусном расстоянии формируют матрицу схожести. Далее
по строкам матрицы схожести вычисляется спектр схожести первого видео со вторым, а по столбцам вычисляется процент схожести 
второго видео с первым. На спектрах ищутся максимальные пики схожести и по ним вычисляются интервалы заимствования видео.
 
## Документация
- [main.py](main_doc.md)
- [ml_service](ml_service_doc.md)

