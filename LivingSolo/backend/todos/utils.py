from tags.utils import get_tag_dict_from_obj_list

NULL_COLOR = '#333333'

# Params [todo_category]: todo_category django object
def get_todo_category_and_color_dict_nullcheck(todo_category):
    if todo_category:
        return {
            "id": todo_category.id,
            "name": todo_category.name,
            "color": todo_category.color,
        }, todo_category.color
    else:
        return {
            "id": -1,
            "name": '삭제된 카테고리',
            "color": NULL_COLOR,
        }, NULL_COLOR


# Return Todo Dictionary from Django Todo Object
def get_todo_dict_from_tag_obj(todo_obj):
    categ_json, categ_color = get_todo_category_and_color_dict_nullcheck(todo_obj.category)
    return {
        "id": todo_obj.id,
        "name": todo_obj.name,
        "tag": get_tag_dict_from_obj_list(list(todo_obj.tag.all().values())),
        "done": todo_obj.done,
        "color": categ_color,
        "category": categ_json,
        "priority": todo_obj.priority,
        "deadline": todo_obj.deadline,
        "is_hard_deadline": todo_obj.is_hard_deadline,
        "period": todo_obj.period,
    }


# Return TodoCategory Dictionary from Django TodoCategory Object
def get_todo_category_dict_from_obj(todo_category_obj):
    return {
        "id": todo_category_obj.id,
        "name": todo_category_obj.name,
        "color": todo_category_obj.color,
        "tag": get_tag_dict_from_obj_list(list(todo_category_obj.tag.all().values())),
    }
