import os
from django import template
from django.templatetags.static import static

register = template.Library()

@register.simple_tag
def list_dir(path_directory):
    list_images_urls = []
    for filename in os.listdir(path_directory):
        file_path = os.path.join(path_directory, filename)
        url = ''
        # file_path example: visualization/static/visualization/img\9.png
        # needs to erase the first two component from the path
        for x in file_path.split('/')[2:]:
            url = os.path.join(url, x)
        list_images_urls.append(static(url))
    return list_images_urls