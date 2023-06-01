import json
import os.path

import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.templatetags.static import static

from visualization.augmenting_traffic_signs.BBox import BBox
from visualization.augmenting_traffic_signs.utils_augmenting import load_model_inpainting, load_model_sign_embed
from visualization.generators_factory.setup_inpainting_generator import setup_inpainting_generator_environment
from visualization.generators_factory.setup_realism_generator import setup_realism_generator_environment
from visualization.upload_form import UploadFileForm
from django.core.files.storage import FileSystemStorage
from ML_Traffic_Visualization_Tool.settings import MEDIA_ROOT, BASE_DIR
from visualization.utils import normalize_image_0_1, read_image_cv2, save_image, normalize_image_01_negative

HEIGHT_INPUT_IMAGE_FRAME = 800
WIDTH_INPUT_IMAGE_FRAME = 800

def visualization(request):
    template = loader.get_template("base.html")
    return HttpResponse(template.render())

def upload_file(request):
    template = loader.get_template("upload.html")
    context = {}

    if request.method == "POST":
        uploaded_file = request.FILES["traffic_image"]

        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)

        absolute_path_image = os.path.join(MEDIA_ROOT, uploaded_file.name)
        image = read_image_cv2(absolute_path_image)

        uploaded_file_url = fs.url(filename)
        filename_image = uploaded_file_url.split('/')[-1]
        ext = filename_image.split('.')[1]
        name_image = filename_image.split('.')[0]

        if ext == 'ppm':
            fs.delete(uploaded_file.name)
            save_image(image, MEDIA_ROOT, name_image + '.png')
            uploaded_file_url = uploaded_file_url.split('.')[0] + '.png'

        height_frame = min(HEIGHT_INPUT_IMAGE_FRAME, image.shape[0])
        width_frame = min(WIDTH_INPUT_IMAGE_FRAME, image.shape[1])

        context['uploaded_file_url'] = uploaded_file_url
        context['height_frame'] = height_frame
        context['width_frame'] = width_frame
        print("_________________uploaded_file uploaded_file uploaded_file uploaded_file: ", uploaded_file)
        print("________Height Frame: ", context['height_frame'])
        print("________Width Frame: ", context['width_frame'])
        print("Absolute path image: ", absolute_path_image)
        print("uploaded_file_url: ", uploaded_file_url)

        print("???????????????????: ", ext)

        # print(image)
    return HttpResponse(template.render(context, request))


def augmenting_traffic_signs(request):
    realism_generator_data = None
    with open(os.path.join(BASE_DIR, r"productionfiles\visualization\RealismGeneratorConfig.json")) as f:
        realism_generator_data = json.load(f)

    inpainting_generator_data = None
    with open(os.path.join(BASE_DIR, r"productionfiles\visualization\InpaintingGeneratorConfig.json")) as f:
        inpainting_generator_data = json.load(f)

    template = loader.get_template('augmenting.html')
    context = {}
    if request.method == "POST":
        my_data = json.loads(request.POST['selections'])
        ruttier_image_name = json.loads(request.POST['upload_image_name'])

        # inpainted_generator = load_model_inpainting()
        # augmenting_sign_generator22 = load_model_sign_embed()

        realism_gen_type = json.loads(request.POST['realism_gen_type'])
        inpaint_gen_type = json.loads(request.POST['inpaint_gen_type'])
        realism_generator_factory = setup_realism_generator_environment(realism_gen_type, realism_generator_data)
        inapainting_generator_factory = setup_inpainting_generator_environment(inpaint_gen_type, inpainting_generator_data)
        inpainted_generator = inapainting_generator_factory.createFactory()
        augmenting_sign_generator = realism_generator_factory.createFactory()

        ruttier_image = read_image_cv2(os.path.join(MEDIA_ROOT, ruttier_image_name))
        ruttier_image = normalize_image_01_negative(ruttier_image)

        # print("__________________My_data: ", my_data)
        # print("__________________My_Image_data: ", ruttier_image_name)
        for rectangle_info in my_data:
            coord_x = int(rectangle_info['coord_x'][:-2])
            coord_y = int(rectangle_info['coord_y'][:-2])
            width = int(rectangle_info['width_rect'][:-2])
            height = int(rectangle_info['height_rect'][:-2])
            sign_image_name = rectangle_info['sign_image_name']
            bbox = BBox(coord_x, coord_y, width, height, sign_image_name, ruttier_image,
                        inpainted_generator, augmenting_sign_generator)
            ruttier_image = bbox.add_new_synthetic_signs()
            # print(coord_x, coord_y, width, height, sign_image_name)
            # print("__________________________________________________________________")

        ruttier_image = normalize_image_0_1(ruttier_image)
        saved_file_name = "replace_" + ruttier_image_name
        save_image(ruttier_image, MEDIA_ROOT, saved_file_name)

        fs = FileSystemStorage()
        saved_replace_image_url = fs.url(saved_file_name)
        context['saved_replace_image_url'] = saved_replace_image_url
        context['saved_file_name'] = saved_file_name
        height_frame = min(1000, ruttier_image.shape[0])
        width_frame = min(1000, ruttier_image.shape[1])
        context['height_frame'] = height_frame
        context['width_frame'] = width_frame
    return HttpResponse(template.render(context, request))

def backend_work(request):
    realism_gen_type = request.POST['realism_gen_type']

    with open(os.path.join(BASE_DIR, r"productionfiles\visualization\RealismGeneratorConfig.json")) as f:
        realism_generator_data = json.load(f)
        allowed_icons = realism_generator_data[realism_gen_type]['list_prototype_signs']

    list_images_urls = []
    path_directory = 'visualization/static/visualization/icons/images'
    for filename in os.listdir(path_directory):
        if filename in allowed_icons:
            file_path = os.path.join(path_directory, filename)
            url = ''
            # file_path example: visualization/static/visualization/img\9.png
            # needs to erase the first two component from the path
            for x in file_path.split('/')[2:]:
                url = os.path.join(url, x)
            list_images_urls.append(static(url))
    return HttpResponse(json.dumps(list_images_urls))

def coords(request):
    template = loader.get_template("coords.html")
    context = {}
    return HttpResponse(template.render(context, request))