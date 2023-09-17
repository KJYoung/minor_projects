# Params [tag_value_list]: list(tag.values())
# ex. list(trxn_elem.tag.values())
def get_tag_dict_from_obj_list(tag_value_list):
    tags = []
    for tag_elem in tag_value_list:
        # {
        #   'id': 1,
        #   'created': datetime.datetime(2023, 7, 12, 22, 43, 56, 23036),
        #   'updated': datetime.datetime(2023, 7, 12, 22, 43, 56, 27034),
        #   'name': 'example1',
        #   'color': '#000000',
        #   'type_class_id': 1
        # }

        tags.append(
            {
                "id": tag_elem['id'],
                "name": tag_elem['name'],
                "color": tag_elem['color'],
            }
        )
    return tags


# Return Tag Dictionary(except tag_class information) from Django Tag Object
def get_tag_brief_dict_from_tag_obj(tag_obj):
    return {
        "id": tag_obj.pk,
        "name": tag_obj.name,
        "color": tag_obj.color,
    }


# Return Tag Dictionary from Django Tag Object
def get_tag_dict_from_tag_obj(tag_obj):
    return {
        "id": tag_obj.pk,
        "name": tag_obj.name,
        "color": tag_obj.color,
        "tag_class": get_tag_class_brief_dict_from_obj(tag_obj.tag_class),
    }


# Return TagPreset Dictionary from Django TagPreset Object
def get_tag_preset_dict_from_obj(tag_preset_obj):
    return {
        "id": tag_preset_obj.id,
        "name": tag_preset_obj.name,
        "tags": get_tag_dict_from_obj_list(list(tag_preset_obj.tags.all().values())),
    }


# Return TagClass Dictionary from Django TagClass Object
def get_tag_class_dict_from_obj(tag_class_obj):
    return {
        "id": tag_class_obj.id,
        "name": tag_class_obj.name,
        "color": tag_class_obj.color,
        "tags": get_tag_dict_from_obj_list(list(tag_class_obj.tag.all().values())),
    }


# Return TagClass Dictionary(except tags) from Django TagClass Object
def get_tag_class_brief_dict_from_obj(tag_class_obj):
    return {
        "id": tag_class_obj.id,
        "name": tag_class_obj.name,
        "color": tag_class_obj.color,
    }
