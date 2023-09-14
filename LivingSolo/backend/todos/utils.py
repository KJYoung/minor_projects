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
