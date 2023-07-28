"""
    Todo, TodoCategory, 할일에 관한 처리 로직입니다.
"""
import json
from json.decoder import JSONDecodeError
from calendar import monthrange

from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, HttpResponseNotFound, JsonResponse
from todos.models import Todo


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
            result[todo_elem.deadline.day].append(
                {
                    "id": todo_elem.id,
                    "name": todo_elem.name,
                    "tag": tags,
                    "done": todo_elem.done,
                    "color": todo_cate.color,
                    "category": {
                        "name": todo_cate.name,
                        "color": todo_cate.color,
                    },
                    "priority": todo_elem.priority,
                    "deadline": todo_elem.deadline,
                    "is_hard_deadline": todo_elem.is_hard_deadline,
                    "period": todo_elem.period,
                }
            )
        return JsonResponse({"elements": result}, safe=False)


@require_http_methods(['PUT'])
def toggle_todo(request, todo_id):
    """
    PUT : toggle done
    """
    if request.method == 'PUT':
        try:
            todo_id = int(todo_id)
            todo_obj = Todo.objects.get(pk=todo_id)

            todo_obj.done = not (todo_obj.done)
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
