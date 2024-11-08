<h1 align="center">OCR-KTP</h1>

<h2 align="center">OCR KTP Project</h2>

**OCR-KTP** is created to extracted the KTP Holder's Name, NIK, Address and Date of Birth.

<h2 style="font-weight: 800;">🚀 How to launch</h2>

**First launch**
```console
$ git clone https://github.com/NicholasAKS438/OCR-KTP.git
$ ./setup_env.bat
$ fastapi dev main.py
```

**Consequent launch**
```console
$ ./call_env.bat
$ fastapi dev main.py
```
---

<h2 style="font-weight: 800;">🚀 API Endpoint</h2>

<h3>/extract-text</h3>
<h4>Input</h4>Form-data with KTP image
<h4>Output</h4>

<p>Successful output</p>

```console
{
  "NIK": "0000000000000000",
  "Nama": "ABC",
  "Tanggal Lahir": "01-02-2000",
  "Alamat": {
    "Alamat": "ABC",
    "RT/RW": "001/003",
    "Kelurahan/Desa": "ABC",
    "Kecamatan": "ABC"
  }
}
```

<p>Null info</p>

```console
{"detail": "Gambar tidak jelas"}
```

<p>KTP not in image</p>

```console
{"detail": "Gambar bukanlah KTP"}
```

<p>Invalid image type</p>

```console
{"detail": "File is not an image!"}
```

<p>Catch all error</p>

```console
{"detail": "Image processing failed"}
```
---
