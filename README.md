# Selldone - Background Removal

This project focuses on background removal using deep learning techniques, with an emphasis on speed and accuracy. Our
goal is to create a readily usable, open-source tool for developers, free of charge. It leverages pre-trained models
available in the open-source community and integrates them with a FastAPI framework for efficient and effective
background removal tasks.

![backgroundremove.jpg](_docs%2Fimages%2Fbackgroundremove.jpg)
![sample-ai-bg-remove.jpg](_docs%2Fimages%2Fsample-ai-bg-remove.jpg)

### Endpoint available

| Endpoint                        | Description                |
|---------------------------------|----------------------------|
| http://localhost:8000/remove-bg | Remove background endpoint |

### Install

1. Clone this repository

```bash
git clone https://github.com/selldone/bg-remove.git

```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download model

Download model from ([**Google Drive**](https://drive.google.com/file/d/1XHIzgTzY5BQHw140EDIgwIb53K659ENH/view?usp=sharing)) and put the `isnet-general-use.pth` file in the `service/isnet/model` folder


4. Start web-application

#### ISNET (NEW)

```bash
uvicorn service.app:app --host localhost --port 8000
```

#### U2NET (OLD)

```bash
uvicorn service.app_u2net:app --host localhost --port 8000
```


### Credit:

https://xuebinqin.github.io/dis/index.html

