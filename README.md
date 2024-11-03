Team AVATECH : 	
		- RABESON Mamy Nirina - 300I22
		- RAMAMONJIARISOA Mihaingo Andiniaina - 232I22
		- RATSIMBAZAFY Lalarijaona Ambinintsoa - 288I22

A.INSTALLATION
	1. Clonez le : 
		- back-end depuis : git clone -b backend_fixed https://github.com/hunter609/ProjetIAL3.git
		- front-end depuis : git clone https://github.com/hunter609/ProjetIAL3Front.git

	2. Naviguez vers le repertoire où se situe le Front-End du projet
		a. Ouvrez cmd sur ce repertoire
		b. executez npm install

	3. Rentrez dans le repertoire où se situe le Back-End du projet
		a. Ouvrez cmd sur ce repertoire
		b. executez : pip install absl-py==2.1.0 aiohappyeyeballs==2.4.3 aiohttp==3.10.10 aiosignal==1.3.1 asgiref==3.7.2 astunparse==1.6.3 attrs==24.2.0 beautifulsoup4==4.12.3 blinker==1.8.2 certifi==2024.2.2 cffi==1.17.1 charset-normalizer==3.3.2 click==8.1.7 colorama==0.4.6 contourpy==1.3.0 cycler==0.12.1 dateparser==1.2.0 Django==4.2.4 django-ckeditor==6.7.0 django-cors-headers==4.2.0 django-environ==0.11.1 django-js-asset==2.1.0 djangorestframework==3.14.0 fire==0.6.0 Flask==3.0.3 Flask-Cors==5.0.0 flatbuffers==24.3.25 fonttools==4.53.1 frozendict==2.4.6 frozenlist==1.4.1 gast==0.6.0 google-pasta==0.2.0 grpcio==1.67.0 h11==0.14.0 h5py==3.12.1 html5lib==1.1 idna==3.6 itsdangerous==2.2.0 Jinja2==3.1.4 joblib==1.4.2 keras==3.6.0 kiwisolver==1.4.7 libclang==18.1.1 lxml==5.3.0 Markdown==3.7 markdown-it-py==3.0.0 MarkupSafe==3.0.2 matplotlib==3.9.2 mdurl==0.1.2 ml-dtypes==0.4.1 multidict==6.1.0 multitasking==0.0.11 namex==0.0.8 numpy==1.26.4 opencv-python-headless==4.10.0.84 opt_einsum==3.4.0 optree==0.13.0 outcome==1.3.0.post0 packaging==24.1 pandas==2.2.3 pdf2docx==0.5.8 peewee==3.17.7 Pillow==10.0.0 platformdirs==4.3.6 propcache==0.2.0 protobuf==4.25.5 psycopg2-binary==2.9.9 pycparser==2.22 pycryptodome==3.21.0 Pygments==2.18.0 PyMuPDF==1.24.10 PyMuPDFb==1.24.10 pyparsing==3.2.0 PySocks==1.7.1 python-binance==1.0.19 python-dateutil==2.8.2 python-docx==1.1.2 pytz==2023.3 regex==2024.9.11 requests==2.31.0 rich==13.9.2 scikit-learn==1.5.2 scipy==1.14.1 selenium==4.25.0 setuptools==75.2.0 six==1.16.0 sniffio==1.3.1 sortedcontainers==2.4.0 soupsieve==2.6 sqlparse==0.4.4 stripe==8.4.0 tensorboard==2.17.1 tensorboard-data-server==0.7.2 tensorflow==2.17.0 tensorflow-intel==2.17.0 termcolor==2.4.0 threadpoolctl==3.5.0 trio==0.27.0 trio-websocket==0.11.1 typing_extensions==4.12.2 tzdata==2024.1 tzlocal==5.2 ujson==5.10.0 urllib3==2.2.1 webencodings==0.5.1 websocket-client==1.8.0 websockets==13.1 Werkzeug==3.0.4 wheel==0.44.0 wrapt==1.16.0 wsproto==1.2.0 yarl==1.15.5 yfinance==0.2.44

	4. Dans le repertoire front-end, executez : npm run dev
	5. Dans le repertoire back-end, executez : python app.py


B.MODE DE FONCTIONNEMENT
	C'est une IA pour prédire le prix de l'or mondial dans un délai de 30 à 60 jours après. Codée en python, l'IA fournit un graphisme qui se base sur des données réelles fournies par YAOOH FINANCE et utilise la regression linéaire pour le traitement des données. Dans le front-end, on peut apercevoir et interagir avec le graphisme fournit par l'IA.
