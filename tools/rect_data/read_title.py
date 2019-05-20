from PIL import Image
import os, sys

import pyocr
import pyocr.builders

# img_files = os.listdir(img_path)


tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print("Will use tool '%s'" % (tool
.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
# lang = "jpn_vert"
# lang = "jpn"
lang = "eng"
print("Will use lang '%s'" % (lang))
print("=================================================")


# img_files = os.listdir("")
rotate_trim = "rotate_trim"
# rotate_trim = "trims"
rotate_trim = "rotate_gray"
dirs = os.listdir("./" + rotate_trim + "/")
for i, dir_name in enumerate(dirs):
    print("------------------------------")
    print(dir_name)
    a = os.path.join("./" + rotate_trim, dir_name)
    files = os.listdir(a)
    for j, file_name in enumerate(files):
        a_file = os.path.join("./" + rotate_trim ,dir_name, file_name)
        print("file_name : " + str(a_file))
        txt = tool.image_to_string(
            Image.open(a_file),
            lang=lang,
            builder=pyocr.builders.TextBuilder(tesseract_layout=6)
        )
        print( txt )
        print("")



# txt = tool.image_to_string(
#     Image.open('iroha.png'),
#     lang="jpn",
#     builder=pyocr.builders.TextBuilder(tesseract_layout=6)
# )
# print( txt )
# txt is a Python string
