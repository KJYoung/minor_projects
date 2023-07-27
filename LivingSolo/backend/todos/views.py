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
            print("only added to...", todo_elem.deadline.day)
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
    # else:  ## post
    #     try:
    #         req_data = json.loads(request.body.decode())
    #         element = Transaction(
    #             memo=req_data["memo"],
    #             amount=req_data["amount"],
    #             period=req_data["period"],
    #             date=req_data["date"],
    #         )
    #         element.save()

    #         # Set tags
    #         tag_bubble_list = req_data["tag"]
    #         for tag_bubble in tag_bubble_list:
    #             tag_elem = Tag.objects.get(pk=tag_bubble["id"])
    #             element.tag.add(tag_elem)
    #     except (KeyError, JSONDecodeError):
    #         return HttpResponseBadRequest()
    #     return JsonResponse({"id": element.id, "memo": element.memo}, status=201)
