attr_dict = {}

attr_dict['single_5'] = ["short hair", "young", "elder", "makeup", "no makeup"]

attr_dict['single_6'] = ["long hair", "short hair", "young", "elder", "makeup", "no makeup"]

attr_dict['ablation_study_4'] = ["black hair", "eyeglasses", "male", "smiling"]

attr_dict['prompt_20'] = ["man", "woman", "child", "arched eyebrows", "smiling", "wearing hat", \
                           "wearing eyeglasses", "wearing necklace", "blonde hair","black hair", \
                           "bangs","gray hair", "wavy hair", "bald head","red lip", "makeup", \
                           "wearing earrings","beard","double chin", "young"]

attr_dict['compare_20'] = ["man", "woman", "child", "arched eyebrows", "smiling", "wearing hat", \
                           "wearing eyeglasses", "wearing necklace", "blonde hair","black hair", \
                           "bangs","gray hair", "wavy hair", "bald head","red lip", "makeup", \
                           "wearing earrings","beard","double chin", "young"]

attr_dict['injection_20'] = ["man", "woman", "child", "arched eyebrows", "smiling", "wearing hat", \
                             "wearing eyeglasses", "wearing necklace", "blonde hair","black hair", \
                             "bangs","gray hair", "wavy hair", "bald head","red lip", "makeup", \
                             "wearing earrings","beard","double chin", "young"]

attr_dict['injection_coco_10'] = ["kite", "toaster", "zebra", "chair", "microwave", "skateboard", "clock",
                                  "umbrella", "broccoli", "bear"]

attr_dict['eval_20'] = ["man", "woman", "child", "arched eyebrows", "smiling", "wearing hat", \
                           "wearing eyeglasses", "wearing necklace", "blonde hair","black hair", \
                           "bangs","gray hair", "wavy hair", "bald head","red lip", "makeup", \
                           "wearing earrings","beard","double chin", "young"]


attr_dict['imagenet_1_USER'] = [
    'basketball'
]

attr_dict['coco_1_USER'] = [
    'airplane'
]


# attr_dict['coco_4_USER'] = [
#     'background',
#     'zebra',
#     'grass',
#     'sky'
# ]

attr_dict['coco_4_USER'] = [
    'background',
    'pizza',
    'table',
    'plate'
]

attr_dict['coco_6_USER'] = [
    'airplane',
    'dining table',
    'giraffe',
    'car',
    'umbrella',
    'sheep'
]

attr_dict['coco_80_USER'] = [
    'background',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]

attr_dict['celebA_3_USER'] = ['male', 'smile', 'eyeglasses']

attr_dict['celebA_2_USER'] = ['male', 'sir']

attr_dict['celebA_40_USER'] = [
    "5 o'clock shadow",
    'arched eyebrows',
    'attractive',
    'bags under eyes',
    'bald',
    'bangs',
    'big lips',
    'big nose',
    'black hair',
    'blond hair',
    'blurry',
    'brown hair',
    'bushy eyebrows',
    'chubby',
    'double chin',
    'eyeglasses',
    'goatee',
    'gray hair',
    'heavy makeup',
    'high cheekbones',
    'male',
    'mouth slightly open',
    'mustache',
    'narrow eyes',
    'no beard',
    'oval face',
    'pale skin',
    'pointy nose',
    'receding hairline',
    'rosy cheeks',
    'sideburns',
    'smiling',
    'straight hair',
    'wavy hair',
    'wearing earrings',
    'wearing hat',
    'wearing lipstick',
    'wearing necklace',
    'wearing necktie',
    'young'
]

attr_dict['celebA_50_USER'] = ['male', 'heavy makeup', 'wearing earrings', 'black hair', 'blond hair',
                             'wearing eyeglasses', 'wearing hat', 'young', 'with beard', 'smiling',
                             'dog', 'cat', 'tiger', 'lion', 'bear', 'elephant', 'giraffe', 'zebra',
                             'wolf', 'fox', 'rabbit', 'squirrel', 'eagle', 'penguin', 'dolphin',
                             'whale', 'snake', 'turtle', 'frog', 'butterfly',
                             'sofa', 'chair', 'bed', 'table', 'closet', 'bookshelf', 'mirror', 'nightstand',
                             'carpet', 'television', 'refrigerator', 'washing machine', 'microwave oven', 'bowl',
                             'plate', 'cup', 'curtain', 'pillow', 'clock', 'computer']

attr_dict['celebA_22_USER'] = ['male', 'heavy makeup', 'wearing earrings', 'black hair', 'blond hair',
                               'wearing eyeglasses', 'wearing hat', 'young', 'with beard', 'smiling',
                               'giraffe', 'zebra', 'penguin', 'sofa', 'chair', 'nightstand', 'carpet', 'refrigerator',
                               'washing machine', 'microwave oven', 'curtain', 'pillow']

# attr_dict['try_70_USER'] = ["long hair", "short hair", "young", "elder", "makeup", "no makeup", "table", "no table", "chair", "no chair", "bookshelf", "no bookshelf", "desk", "no desk", "mirror", "no mirror", "bed", "no bed", "lamp", "no lamp", "TV", "no TV", "washing machine", "no washing machine", "telephone", "no telephone",
#                             "dolphin", "no dolphin", "cat", "no cat", "dog", "no dog", "shells", "no shells", "shark", "no shark", "whale", "no whale", "jellyfish", "no jellyfish",
#                             "dragonfly", "no dragonfly", "bee", "no bee", "beetle", "no beetle", "butterfly", "no butterfly", "ant", "no ant", "tiger", "no tiger", "lion", "no lion", "cow", "no cow", "chicken", "no chicken",
#                             "fox", "no fox", "mouse", "no mouse", "rat", "no rat", "bear", "no bear", "owl", "no owl", "swan", "no swan"]
# attr_dict['try_35_USER'] = ["long hair", "young", "makeup", "table", "chair",  "bookshelf",  "desk", "mirror", "bed", "lamp",  "TV", "washing machine", "telephone",
#                             "dolphin", "cat", "dog", "shells", "shark", "whale", "jellyfish",
#                             "dragonfly", "bee",  "beetle",  "butterfly", "ant", "tiger", "lion", "cow", "chicken",
#                             "fox", "mouse", "rat", "bear",  "owl", "swan"]

attr_dict['try_6_USER'] = ["long hair", "short hair", "young", "elder", "makeup", "no makeup"]

attr_dict['ffhq_20_USER'] = ["man", "woman", "child", "arched eyebrows", "smiling", "wearing hat", "wearing eyeglasses", "wearing necklace", "blonde hair","black hair", \
                             "bangs","gray hair", "wavy hair", "bald head","red lip", "makeup", "wearing earlings","beard","double chin", "young"]
                       
attr_dict['ffhq_20_GPT'] = ['clean-shaven', 'beard', 'mustache', 'Wide-eyed', 'thin lips','bald', 'glasses-wearing', 'Freckled', 'almond-shaped eyes', 'Scarred',
                            'Wrinkled', 'soul patch', 'high forehead','hooded eyes', 'piercings', 'prominent cheekbones', 'full lips', 'braided','Upturned-nosed', 'Youthful']


attr_dict['celebA_20_USER'] = ["man", "woman", "child", "arched eyebrows", "smiling", "wearing hat", "wearing eyeglasses", "wearing necklace", "blonde hair","black hair", \
                             "bangs","gray hair", "wavy hair", "bald head","red lip", "makeup", "wearing earlings","beard","double chin", "young"]

# attr_dict['celebA_40_USER'] = ["man", "not man", "woman", "not woman", "child", "not child", "arched eyebrows", "not arched eyebrows", "smiling", "not smiling", "wearing hat",
#                                "not wearing hat", "wearing eyeglasses", "not wearing eyeglasses", "wearing necklace", "not wearing necklace", "blonde hair", "not blonde hair",
#                                "black hair", "not black hair", "bangs", "no bangs", "gray hair", "not gray hair", "wavy hair", "not wavy hair", "bald head", "not bald head",
#                                "red lip", "not red lip", "makeup", "no makeup", "wearing earlings", "not wearing earlings", "beard", "no beard", "double chin", "not double chin",
#                                "young", "not young"]

attr_dict['lsuncat_20_USER'] = ['green eyes', 'blue eyes', 'pink nose', 'black fur', 'white fur', 'yellow eyes', 'fluffy fur', 'solid color', 'black nose',  'tabby pattern',
                                'white paws', 'gray fur', 'brown fur',   'round eyes', 'calico pattern', 'orange fur', 'pointed ears', 'tufted ears', 'slanted eyes', 'long tail']

attr_dict['lsuncat_20_color'] = ['green fur', 'blue fur', 'white fur', 'black fur', 'yellow fur', 'dotted fur', 'ginger fur', 'orange fur', 'tabby fur', 'siamese fur', 'tortoiseshell fur',
                                 'calcico fur', 'cream fur', 'fawn fur', 'lilac fur', 'chocolate fur', 'red fur', 'sable fur', 'striped fur', 'navy fur']

attr_dict['lsuncat_20_shape'] = ['fluffy fur',  'long eyelashes', 'round eyes',  'pointed ears', 'tufted ears', 'slanted eyes', 'long tail',
                                 'long fur',  'short fur', 'small ears', 'small nose','Almond-shaped eyes','round face', 'floppy ears','long whiskers', 'white chin', 'amber eyes',
                                 'hazel eyes', 'Wide-set eyes',  'short tail']

attr_dict['coco_20_USER']=['man', 'woman', 'he', 'people', 'person', 'table', 'group', 'street', 'water', 'plate', 
                           'cat', 'field', 'couple', 'dog', 'food', 'beach', 'bed', 'bathroom', 'pizza', 'grass']

attr_dict['coco_30_USER']=['man', 'woman', 'he', 'people', 'person', 'table', 'group', 'street', 'water', 'plate', 
                           'cat', 'field', 'couple', 'dog', 'food', 'beach', 'bed', 'bathroom', 'pizza', 'grass',
                           'kitchen', 'skateboard', 'picture', 'road', 'train', 'building', 'snow', 'surfboard', 'toilet', 'giraffe']

attr_dict['coco_40_USER']=['man', 'woman', 'he', 'people', 'person', 'table', 'group', 'street', 'water', 'plate', 
                           'cat', 'field', 'couple', 'dog', 'food', 'beach', 'bed', 'bathroom', 'pizza', 'grass',
                           'kitchen', 'skateboard', 'picture', 'road', 'train', 'building', 'snow', 'surfboard', 'toilet', 'giraffe',
                           'room', 'men', 'bunch', 'ball', 'air', 'bench', 'clock', 'boy', 'sign', 'tree']
