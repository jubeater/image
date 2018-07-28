import hashlib
import os
import pathlib
import imghdr
from PIL import Image

def verify(path):
    try:
        image = Image.open(path)
        image.load()
    except :
        os.remove(path)
        print('remove the broken file %s', path)
        pass



def delete(hash):
	current_dir = pathlib.Path(__file__).parent
	pathlist = current_dir.glob('*/*/*.*')
	for path in pathlist:
		if str(path).lower().endswith(('.jpg', '.gif', '.png', '.jpeg')):
			md5 = hashlib.md5(open(str(path),'rb').read()).hexdigest()
			if md5 == hash:
				os.remove(str(path))
				print('removed %s', str(path))
		else:
			os.remove(str(path))
			print('not a image file, removed %s', str(path))
	print('all done.')


def delete_broken():
	current_dir = pathlib.Path(__file__).parent
	pathlist = current_dir.glob('*/*/*.*')
	for path in pathlist:
		verify(str(path))
	print('check done')	


def convert():
    current_dir = pathlib.Path(__file__).parent
    pathlist = current_dir.glob('*/*.*')
    for path in pathlist:
        if str(path).lower().endswith(('.gif', '.png')):
            file, ext = str(path).split('.')
            im = Image.open(path)
            rgb_im = im.convert('RGB')
            rgb_im.save(file + '.jpg')
            print('has convert the' + str(path) + 'to the jpg format!')


def write():
    current_dir = pathlib.Path(__file__).parent
    pathlist = current_dir.glob('*/*.*')
    file = open(str(current_dir) + '/filelist','w', encoding="utf-8")
    for path in pathlist:
        if str(path).lower().endswith(('.jpg', '.jpeg')):
            print(str(path))
            loc = str(path).rfind('/')
            file.write(str(path)+' ' + str(path)[loc - 1] + '\n')
    file.close()
    print('done!')


if __name__ == "__main__":
    write()


