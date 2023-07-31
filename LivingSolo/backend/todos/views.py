"""
    Todo, TodoCategory, 할일에 관한 처리 로직입니다.
"""
import json
from json.decoder import JSONDecodeError
from calendar import monthrange

from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponseNotFound, JsonResponse
from todos.models import Todo, TodoCatecory
from tags.models import Tag

NULL_COLOR = '#333333'


@require_http_methods(['GET', 'POST'])
def general_todo(request):
    """
    GET : get todo list
    POST : create todo element
    """
    if request.method == 'GET':
        # Params: year<Number>, month<Number>

        query_args = {}
        query_args["year"] = request.GET.get("year", None)
        query_args["month"] = request.GET.get("month", None)

        todo_all = Todo.objects.all()
        # Filtering
        # 1. Filter by Year & Month
        if query_args["year"] and query_args["month"]:
            filtered_todo = todo_all.filter(
                deadline__year=query_args["year"], deadline__month=query_args["month"]
            )
        else:
            return HttpResponseBadRequest()

        # result = [[]] *(monthrange(int(query_args["year"]), int(query_args["month"]))[1] + 1)
        # 위와 같이 하면 ex) result[27].append(~~) 했을 때, 모든 Array에 append가 된다. 즉 각 element의 array reference가 같아진다.
        result = [
            []
            for x in range((monthrange(int(query_args["year"]), int(query_args["month"]))[1] + 1))
        ]
        for todo_elem in filtered_todo:
            tags = []
            for tag_elem in list(todo_elem.tag.all().values()):
                tags.append(
                    {
                        "id": tag_elem['id'],
                        "name": tag_elem['name'],
                        "color": tag_elem['color'],
                    }
                )
            todo_cate = todo_elem.category
            categ_json = (
                {
                    "id": todo_cate.id,
                    "name": todo_cate.name,
                    "color": todo_cate.color,
                }
                if todo_cate
                else {
                    "id": -1,
                    "name": '삭제된 카테고리',
                    "color": NULL_COLOR,
                }
            )
            result[todo_elem.deadline.day].append(
                {
                    "id": todo_elem.id,
                    "name": todo_elem.name,
                    "tag": tags,
                    "done": todo_elem.done,
                    "color": todo_cate.color if todo_cate else NULL_COLOR,
                    "category": categ_json,
                    "priority": todo_elem.priority,
                    "deadline": todo_elem.deadline,
                    "is_hard_deadline": todo_elem.is_hard_deadline,
                    "period": todo_elem.period,
                }
            )
        return JsonResponse({"elements": result}, safe=False)
    else:  # POST
        try:
            req_data = json.loads(request.body.decode())
            element = Todo(
                name=req_data["name"],
                priority=req_data["priority"],
                period=req_data["period"],
                deadline=req_data["deadline"],
            )
            element.category = TodoCatecory.objects.get(pk=req_data["category"])
            element.save()

            # Set tags
            tag_bubble_list = req_data["tag"]
            for tag_bubble in tag_bubble_list:
                tag_elem = Tag.objects.get(pk=tag_bubble["id"])
                element.tag.add(tag_elem)
        except (KeyError, JSONDecodeError):
            return HttpResponseBadRequest()
        return JsonResponse({"id": element.id, "name": element.name}, status=201)


@require_http_methods(['POST'])
def duplicate_todo(request):
    """
    POST : duplicate todo element
    """
    try:
        req_data = json.loads(request.body.decode())

        prev_todo = Todo.objects.get(pk=req_data["todo_id"])
        tags = []
        for tag_elem in list(prev_todo.tag.all().values()):
            tags.append(Tag.objects.get(pk=tag_elem['id']))
        prev_todo.pk = None
        prev_todo.save()

        for tag in tags:
            prev_todo.tag.add(tag)

    except (KeyError, JSONDecodeError):
        return HttpResponseBadRequest()
    return JsonResponse({"id": prev_todo.id, "name": prev_todo.name}, status=201)


@require_http_methods(['PUT', 'DELETE'])
def detail_todo(request, todo_id):
    """
    PUT : edit todo's content
    DELETE : delete todo
    """
    if request.method == 'PUT':
        try:
            data = json.loads(request.body.decode())
            todo_id = int(todo_id)
            todo_obj = Todo.objects.get(pk=todo_id)

            todo_obj.name = data["name"]
            todo_obj.save()
            return JsonResponse({"message": "success"}, status=200)
        except Todo.DoesNotExist:
            return HttpResponseNotFound()
    else:  ## delete
        try:
            todo_id = int(todo_id)
            todo_obj = Todo.objects.get(pk=todo_id)

            todo_obj.delete()
            return JsonResponse({"message": "success"}, status=200)
        except Todo.DoesNotExist:
            return HttpResponseNotFound()


@require_http_methods(['PUT'])
def toggle_todo(request, todo_id):
    """
    PUT : toggle done
    """
    if request.method == 'PUT':
        try:
            todo_id = int(todo_id)
            todo_obj = Todo.objects.get(pk=todo_id)

            todo_obj.done = not todo_obj.done
            todo_obj.save()
            return JsonResponse({"message": "success"}, status=200)
        except Todo.DoesNotExist:
            return HttpResponseNotFound()
    else:  ## delete
        try:
            todo_id = int(todo_id)
            todo_obj = Todo.objects.get(pk=todo_id)
            todo_obj.delete()
            return JsonResponse({"message": "success"}, status=200)
        except Todo.DoesNotExist:
            return HttpResponseNotFound()


## TodoCategory
@require_http_methods(['GET', 'POST'])
def general_todo_category(request):
    """
    GET : get todo category list
    """
    if request.method == 'GET':
        try:
            result = []
            for tc_elem in TodoCatecory.objects.all():
                tags = []
                for tag_elem in list(tc_elem.tag.all().values()):
                    tags.append(
                        {
                            "id": tag_elem['id'],
                            "name": tag_elem['name'],
                            "color": tag_elem['color'],
                        }
                    )
                result.append(
                    {
                        "id": tc_elem.id,
                        "name": tc_elem.name,
                        "color": tc_elem.color,
                        "tag": tags,
                    }
                )
            return JsonResponse({"elements": result}, safe=False)
        except (KeyError, JSONDecodeError, TodoCatecory.DoesNotExist):
            print("ERROR from general_todo_category")
            return HttpResponseBadRequest()
    else:  # POST
        try:
            req_data = json.loads(request.body.decode())
            element = TodoCatecory(
                name=req_data["name"],
                color=req_data["color"],
            )
            element.save()

            # Set tags
            tag_bubble_list = req_data["tag"]
            for tag_bubble in tag_bubble_list:
                tag_elem = Tag.objects.get(pk=tag_bubble["id"])
                element.tag.add(tag_elem)
        except (KeyError, JSONDecodeError):
            return HttpResponseBadRequest()
        return JsonResponse({"id": element.id, "name": element.name}, status=201)


@require_http_methods(['PUT', 'DELETE'])
def detail_todo_category(request, categ_id):
    """
    PUT : edit category's content
    DELETE : delete category
    """
    if request.method == 'PUT':
        try:
            data = json.loads(request.body.decode())
            categ_id = int(categ_id)
            categ_obj = TodoCatecory.objects.get(pk=categ_id)

            categ_obj.name = data["name"]
            categ_obj.amount = data["amount"]
            categ_obj.save()
            return JsonResponse({"message": "success"}, status=200)
        except TodoCatecory.DoesNotExist:
            return HttpResponseNotFound()
    else:  ## delete
        try:
            categ_id = int(categ_id)
            categ_obj = TodoCatecory.objects.get(pk=categ_id)

            categ_obj.delete()
            return JsonResponse({"message": "success"}, status=200)
        except TodoCatecory.DoesNotExist:
            return HttpResponseNotFound()
