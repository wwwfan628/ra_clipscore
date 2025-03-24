attr_dict = {}

attr_dict['celebA_50_USER'] = ['male', 'heavy makeup', 'wearing earrings', 'black hair', 'blond hair',
                             'wearing eyeglasses', 'wearing hat', 'young', 'with beard', 'smiling',
                             'dog', 'cat', 'tiger', 'lion', 'bear', 'elephant', 'giraffe', 'zebra',
                             'wolf', 'fox', 'rabbit', 'squirrel', 'eagle', 'penguin', 'dolphin',
                             'whale', 'snake', 'turtle', 'frog', 'butterfly',
                             'sofa', 'chair', 'bed', 'table', 'closet', 'bookshelf', 'mirror', 'nightstand',
                             'carpet', 'television', 'refrigerator', 'washing machine', 'microwave oven', 'bowl',
                             'plate', 'cup', 'curtain', 'pillow', 'clock', 'computer']

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

attr_dict['celebA_40_USER'] = ["man", "not man", "woman", "not woman", "child", "not child", "arched eyebrows", "not arched eyebrows", "smiling", "not smiling", "wearing hat",
                               "not wearing hat", "wearing eyeglasses", "not wearing eyeglasses", "wearing necklace", "not wearing necklace", "blonde hair", "not blonde hair",
                               "black hair", "not black hair", "bangs", "no bangs", "gray hair", "not gray hair", "wavy hair", "not wavy hair", "bald head", "not bald head",
                               "red lip", "not red lip", "makeup", "no makeup", "wearing earlings", "not wearing earlings", "beard", "no beard", "double chin", "not double chin",
                               "young", "not young"]

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
