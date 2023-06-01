import json
import os

from django import template
from django.templatetags.static import static

from ML_Traffic_Visualization_Tool.settings import BASE_DIR

register = template.Library()

@register.simple_tag
def extract_type_realism_gens(config_file):
    list_types = []
    with open(os.path.join(BASE_DIR, config_file)) as f:
        realism_generator_data = json.load(f)
        for type_gen in realism_generator_data.keys():
            list_types.append(type_gen)
    return list_types