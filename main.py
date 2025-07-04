import os
import glob
import tensorflow as tf
import io
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(os.path.join(path, '*.xml')):  # Ensure directory path
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size/width').text),
                     int(root.find('size/height').text),
                     member.find('name').text,
                     int(member.find('bndbox/xmin').text),
                     int(member.find('bndbox/ymin').text),
                     int(member.find('bndbox/xmax').text),
                     int(member.find('bndbox/ymax').text)
                     )
            xml_list.append(value)
    
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# Define class label mapping
class_dict = {'gun': 1, 'asus': 2}  # Modify based on your dataset

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def class_text_to_int(row_label):
    return class_dict.get(row_label.lower(), None)  # Case-insensitive lookup

def create_tf_example(row, path):
    img_path = os.path.join(path, row['filename'])

    if not os.path.exists(img_path):  
        print(f"⚠️ Image not found: {img_path}, skipping...")
        return None

    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_image = fid.read()
    image = Image.open(io.BytesIO(encoded_image))
    width, height = image.size

    filename = row['filename'].encode('utf8')
    
    # Check the image format and set the format accordingly
    if row['filename'].lower().endswith('.png'):
        image_format = b'png'
    elif row['filename'].lower().endswith('.jpg') or row['filename'].lower().endswith('.jpeg'):
        image_format = b'jpeg'
    else:
        print(f"⚠️ Unsupported image format for {row['filename']}, skipping...")
        return None

    xmins = [row['xmin'] / width]
    xmaxs = [row['xmax'] / width]
    ymins = [row['ymin'] / height]
    ymaxs = [row['ymax'] / height]
    
    class_id = class_text_to_int(row['class'])
    if class_id is None:
        print(f"⚠️ Skipping {row['filename']} due to unknown class: {row['class']}")
        return None

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_image),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_feature(row['class'].encode('utf8')),
        'image/object/class/label': int64_feature(class_id),
    }))
    return tf_example

def generate_tfrecord(csv_input, image_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    examples = pd.read_csv(csv_input)

    total_images = len(examples)
    print(f"📂 Found {total_images} entries in CSV.")

    processed = 0
    for index, row in examples.iterrows():
        tf_example = create_tf_example(row, image_dir)
        if tf_example:
            writer.write(tf_example.SerializeToString())
            processed += 1

    writer.close()
    print(f'✅ Successfully created TFRecord at {output_path} ({processed}/{total_images} processed)')

if __name__ == "__main__":

    xml_folder = os.path.join(os.getcwd(), 'Inputs')  # Set to the folder containing XML files
    xml_df = xml_to_csv(xml_folder)
    xml_df.to_csv('annotations.csv', index=False)
    print('✅ Successfully converted XML to CSV.')

    annotations_file = os.path.join(os.getcwd(), 'annotations.csv')
    images_folder = os.path.join(os.getcwd(), 'Inputs')   # Ensure this contains images
    output_tfrecord = 'train.record'

    generate_tfrecord(annotations_file, images_folder, output_tfrecord)

    # ✅ Verify the output using TensorFlow 2.x compatible method
    print("\n🔍 Verifying TFRecord contents:")
    dataset = tf.data.TFRecordDataset(output_tfrecord)
    for i, raw_record in enumerate(dataset.take(5000)):  # Read first 3 records
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print(f"📝 Example {i+1}:")

        # Filename
        print("Filename:", example.features.feature['image/filename'].bytes_list.value[0].decode('utf-8'))

        # Image Dimensions
        print("Image dimensions:", example.features.feature['image/width'].int64_list.value[0], "x",
              example.features.feature['image/height'].int64_list.value[0])
        
        # Bounding Box Coordinates
        print("Bounding Box:")
        print("  xmin:", example.features.feature['image/object/bbox/xmin'].float_list.value[0])
        print("  ymin:", example.features.feature['image/object/bbox/ymin'].float_list.value[0])
        print("  xmax:", example.features.feature['image/object/bbox/xmax'].float_list.value[0])
        print("  ymax:", example.features.feature['image/object/bbox/ymax'].float_list.value[0])
        
        # Class Label
        print("Class label:", example.features.feature['image/object/class/text'].bytes_list.value[0].decode('utf-8'))
        print("Class ID:", example.features.feature['image/object/class/label'].int64_list.value[0])
        
        # Image Encoding Format
        print("Image Format:", example.features.feature['image/format'].bytes_list.value[0].decode('utf-8'))
        
        # Source ID
        print("Source ID:", example.features.feature['image/source_id'].bytes_list.value[0].decode('utf-8'))
        
        print("\n" + "="*50 + "\n")  # Separator for readability



