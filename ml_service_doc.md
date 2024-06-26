# Документация по методу REST
[Назад в README](README.md)
## Описание веб-сервиса
Эндпоинт **/set_video_download** используется для загрузки видеофайла по указанному URL, проверки его целостности с помощью 
MD5-хэш суммы, а затем для выполнения одной из двух операций: индексирования видео в базе данных или поиска видео 
в базе данных. \
 **/task_status:** Позволяет проверить статус выполнения задачи по её идентификатору.
### URL
/set_video_download \
/task_status



### Метод
POST

### Параметры запроса (JSON)
- download_url (string): URL для загрузки видеофайла. **Обязательный параметр**.
- filename (string): Имя файла для сохранения загруженного видео. **Обязательный параметр**.
- md5 (string): MD5-хэш сумма для проверки целостности загруженного файла. **Обязательный параметр**.
- purpose (string): Цель операции, может быть либо "index", либо "val". **Обязательный параметр**.
#### Примеры запроса  /set_video_download

{\
    "download_url": "http://example.com/video.mp4", \
    "filename": "video.mp4",\
    "md5": "d41d8cd98f00b204e9800998ecf8427e",\
    "purpose": "index"\
}
#### Ответы /set_video_download

  - **200 OK**
    - Для purpose = "index":
    {
          "task_id": "generated_task_id"
      }
      
    - Для purpose = "val": {
          "task_id": "generated_task_id"
      }
  - **422 Unprocessable Entity**
    - Если download_url пуст: {
          "error": "download url cant be empty"
      }
   
    - Если purpose не равен "index" или "val":   {
          "error": "purpose can be either index or val"
      }
  - **500 Internal Server Error**
    - Если MD5-хэш сумма не совпадает:{
          "error": "file integrity cant be verified"
      }
  #### Пример использования
```python
import requests

url = 'http://yourserver.com/set_video_download'
data = {
    "download_url": "http://example.com/video.mp4",
    "filename": "video.mp4",
    "md5": "d41d8cd98f00b204e9800998ecf8427e",
    "purpose": "index"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Success:", response.json())
else:
    print("Error:", response.json())
```
#### Ответы /task_status
- **200 OK**
  { \
      "id": "generated_task_id", \
      "type": "index_in_db", \
      "status": "ready", \
      "content": {"indexed": True} \
  }
- **422 Unprocessable Entity**
    { \
        "result": "error", \
        "error_str": "task_id cant be empty" \
    }
    
## Важные моменты

1. **Проверка входных данных**: Эндпоинт проверяет, что download_url не пуст, а purpose имеет допустимое значение.
2. **Целостность файла**: Проверка MD5-хэш суммы загруженного файла для подтверждения его целостности.
3. **Операции**: В зависимости от значения purpose, выполняется либо индексирование видео в базе данных, либо поиск в базе данных.
4. **Ошибки**: Возвращаются соответствующие коды и сообщения об ошибках в случае некорректного запроса или проблем при выполнении операций.

Этот эндпоинт полезен для автоматизации загрузки и обработки видеофайлов с последующим их индексированием или поиском в базе данных.

## Скрипты
### app.py
Инициализация flusk
### db_utils.py
Методы для работы с векторной базой данных: создать таблицу, получить базу данных, добавить вектор
### embeddings.py
Методы для работы с эмбеддингами: получить эмбеддинги видео, получить эмбеддинги аудио
### indexer.py
Методы для индексирования видео в базу данных. Получает векторное представление видео и аудио, сохраняет в векторную базу данных lanceDB. Сохраняет информацию об индексированных файлах в indexed_files.json файл
### peaks.py
Методы для поиска пиков схожести в матрице схожести видеофайлов, вычисления интервалов заимствования видео
### web_inference.py
Методы для веб-инференса: загрузка видеофайла, проверка целостности, индексирование видео, поиск видео
