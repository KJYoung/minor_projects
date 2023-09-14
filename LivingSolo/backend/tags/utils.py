# Params [tag_value_list]: list(tag.values())
# ex. list(trxn_elem.tag.values())
def get_tag_dict_from_obj(tag_value_list):
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
