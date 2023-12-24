import tensorflow as tf

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def load_labels(label_path):
    labels = []
    boxes = []
    with open(label_path) as fp:
      for line in fp.readlines():
        line = line.strip("\n")
        lbl, cx, cy, w, h = line.split(" ")
        labels.append(int(lbl))
        boxes.append([float(cx), float(cy), float(w), float(h)])

    return boxes, labels

def load_dataset(image_path, classes, bbox):
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }

    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

def get_dataset(images_path, labels_path):
    image_paths = []
    bbox = []
    classes = []
    for filename in tqdm(os.listdir(images_path)):
        fname = filename[:-3]
        try:
            boxes, class_ids = load_labels(os.path.join(labels_path, fname+"txt"))
            image_paths.append(os.path.join(images_path, filename))
            classes.append(class_ids)
            bbox.append(boxes)
        except FileNotFoundError as e:
            print(e)
            continue

    bbox = tf.ragged.constant(bbox)
    classes = tf.ragged.constant(classes)
    image_paths = tf.ragged.constant(image_paths)

    data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

    return data

def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]

    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )

def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]
