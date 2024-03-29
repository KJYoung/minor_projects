"""
    Todo, TodoCategory, 할일에 관한 처리 로직입니다.
"""
from datetime import timedelta
import json
from json.decoder import JSONDecodeError
from calendar import monthrange

from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponseNotFound, JsonResponse
from todos.models import Todo, TodoCategory
from tags.models import Tag
from todos.utils import get_todo_dict_from_tag_obj, get_todo_category_dict_from_obj


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
            result[todo_elem.deadline.day].append(get_todo_dict_from_tag_obj(todo_elem))

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
            element.category = TodoCategory.objects.get(pk=req_data["category"])
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

        prev_tag = prev_todo.tag.all()
        prev_todo.pk = None
        prev_todo.done = False  # Maybe Duplicated Jobs to be Done Again.
        prev_todo.save()
        prev_todo.tag.set(prev_tag)

        return JsonResponse({"message": "success"}, status=200)
    except (KeyError, JSONDecodeError):
        return HttpResponseBadRequest()


@require_http_methods(['POST'])
def dup_again_todo(request):
    """
    POST : duplicate todo element [with changed date]
    """
    try:
        req_data = json.loads(request.body.decode())

        prev_todo = Todo.objects.get(pk=req_data["todo_id"])

        prev_tag = prev_todo.tag.all()
        prev_todo.pk = None
        prev_todo.done = False  # Maybe Duplicated Jobs to be Done Again.
        prev_todo.deadline = req_data["date"]
        prev_todo.save()
        prev_todo.tag.set(prev_tag)

        return JsonResponse({"message": "success"}, status=200)
    except (KeyError, JSONDecodeError):
        return HttpResponseBadRequest()


@require_http_methods(['POST'])
def postpone_todo(request):
    """
    POST : postpone todo element
    """
    try:
        req_data = json.loads(request.body.decode())

        target_todos = Todo.objects.filter(deadline=req_data["date"], done=False)

        for target_todo in target_todos:
            target_todo.deadline = target_todo.deadline + timedelta(days=req_data["postponeDayNum"])
            target_todo.save()

        return JsonResponse({"message": "success"}, status=200)
    except (KeyError, JSONDecodeError):
        return HttpResponseBadRequest()


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
            todo_obj.category = TodoCategory.objects.get(pk=data["category"])
            todo_obj.priority = data["priority"]
            todo_obj.deadline = data["deadline"]
            todo_obj.is_hard_deadline = data["is_hard_deadline"]
            todo_obj.period = data["period"]

            todo_obj.save()

            tag_list = data["tag"]
            tag_obj_list = []
            for tag_bubble in tag_list:
                tag_obj_list.append(Tag.objects.get(pk=tag_bubble["id"]))
            todo_obj.tag.set(tag_obj_list)

            return JsonResponse({"message": "success"}, status=200)
        except (Todo.DoesNotExist, TodoCategory.DoesNotExist):
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
    try:
        todo_id = int(todo_id)
        todo_obj = Todo.objects.get(pk=todo_id)

        todo_obj.done = not todo_obj.done
        todo_obj.save()
        return JsonResponse({"message": "success"}, status=200)
    except Todo.DoesNotExist:
        return HttpResponseNotFound()


## TodoCategory
@require_http_methods(['GET', 'POST'])
def general_todo_category(request):
    """
    GET : get todo category list
    POST : create todo category
    """
    if request.method == 'GET':
        try:
            result = [
                get_todo_category_dict_from_obj(tc_elem) for tc_elem in TodoCategory.objects.all()
            ]
            return JsonResponse({"elements": result}, safe=False)
        except (KeyError, JSONDecodeError, TodoCategory.DoesNotExist):
            print("ERROR from general_todo_category")
            return HttpResponseBadRequest()
    else:  # POST
        try:
            req_data = json.loads(request.body.decode())
            element = TodoCategory(
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
            categ_obj = TodoCategory.objects.get(pk=categ_id)

            categ_obj.name = data["name"]
            categ_obj.color = data["color"]
            categ_obj.save()

            tag_list = data["tag"]
            tag_obj_list = []
            for tag_bubble in tag_list:
                tag_obj_list.append(Tag.objects.get(pk=tag_bubble["id"]))
            categ_obj.tag.set(tag_obj_list)

            return JsonResponse({"message": "success"}, status=200)
        except TodoCategory.DoesNotExist:
            return HttpResponseNotFound()
    else:  ## delete
        try:
            categ_id = int(categ_id)
            categ_obj = TodoCategory.objects.get(pk=categ_id)

            categ_obj.delete()
            return JsonResponse({"message": "success"}, status=200)
        except TodoCategory.DoesNotExist:
            return HttpResponseNotFound()
