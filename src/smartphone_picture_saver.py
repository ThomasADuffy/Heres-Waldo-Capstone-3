from PIL import Image, ExifTags
import argparse


def rotate_save(input_image, output_path):
    ''' This is in the flask app, made locally for testing basically takes the
    orentation and resaves it to get the actual representation of the image
    correctly. Used for local testing and work

    Arguments:

    input_image = the input image path
    output_path = this is the output path of the saved image'''
    try:
        image = Image.open(input_image)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        image.save(output_path)
        image.close()
    except (AttributeError, KeyError, IndexError):
        image.save(output_path)
        image.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='remove EXIF tags (orentation mainly)')
    parser.add_argument('path', type=str,
                        help='Location of where the files goes')
    args = parser.parse_args()
    args.path
    args.path.strip('.jpg')+' converted.jpg'
    rotate_save(args.path, '..'+args.path.strip('.jpg')+'converted.jpg')
