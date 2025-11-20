# Kazakh Speech-to-Text Fine-Tune Project

Бұл жоба қазақ тілінде сөйлеуді тану (Speech-to-Text) үшін **Wav2Vec2** моделін fine-tune жасауға арналған.

# Құрылым
audioproject/
├── dataset/
│ ├── train/ ← 320 wav + 320 txt
│ └── testt/ ← 80 wav + 80 txt
├── train.py ← Fine-tune коды
├── infer.py ← Жаңа аудионы тану үшін код
├── requirements.txt ← Барлық кітапханалар
└── README.md ← Бұл файл

- **train/** – модельді үйрету үшін аудио + транскрипция
- **test/** – тестілік аудио + транскрипция
- **train.py** – fine-tune жасау коды
- **infer.py** – жаңа аудионы тану
- **requirements.txt** – барлық қажетті Python кітапханалары

---

# Орнату

1. Python ≥ 3.9 орнатыңыз
2. Репозиторийді клонирлеу немесе жүктеу
3. Қажетті пакеттерді орнату:
```bash
pip install -r requirements.txt

python train.py
Модель audio_dataset/train бойынша үйренеді

Бағалау audio_dataset/test арқылы жүзеге асады

Fine-tune аяқталған соң модель ./wav2vec2-kz ішіне сақталады

WER нәтижесі wer_scores.txt файлына жазылады

# Нәтиже
Fine-tune нәтижесінде WER айтарлықтай төмендейді

Модель қазақша сөйлеуді жақсы таниды

Жаңа аудио немесе телефоннан жазылған файлдар бойынша мәтін алуға дайын

#Кеңестер
GPU болған жағдайда fp16=True қосу арқылы оқытуды жылдамдатуға болады

Epoch санын көбейту арқылы сапасын жақсартуға болады

Үлкен dataset болса, batch size + gradient accumulation қолданып fine-tune жасауға болады

Автор
Nurzhigit Zhumabek