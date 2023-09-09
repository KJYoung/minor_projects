"""
    Tag, 각종 모델에 사용되는 태그에 관한 처리 로직입니다.
"""

import json
from json.decoder import JSONDecodeError
from django.views.decorators.http import require_http_methods
from django.http import HttpResponseBadRequest, JsonResponse
from tags.models import TagClass, Tag

## TagClass
@require_http_methods(['GET', 'POST'])
def general_tag_class(request):
    """
    GET : get tag class list
    POST : create tag category(tag class)
    """
    if request.method == 'GET':
        try:
            result = []
            for trc_elem in TagClass.objects.all():
                tags_list = []
                for tag in list(trc_elem.tag.all().values()):
                    # {'id': 1, 'created': datetime.datetime(2023, 7, 12, 22, 43, 56, 23036), 'updated': datetime.datetime(2023, 7, 12, 22, 43, 56, 27034), 'name': '클1타1', 'color': '#000000', 'type_class_id': 1}
                    tags_list.append({"id": tag['id'], "name": tag['name'], "color": tag['color']})
                result.append(
                    {
                        "id": trc_elem.id,
                        "name": trc_elem.name,
                        "color": trc_elem.color,
                        "tags": tags_list,
                    }
                )
            return JsonResponse({"elements": result}, safe=False)
        except (TagClass.DoesNotExist):
            print("ERROR from general_tag_class")
            return HttpResponseBadRequest()
    else:  # POST REQUEST
        try:
            req_data = json.loads(request.body.decode())
            element = TagClass(
                name=req_data["name"],
                color=req_data["color"],
            )
            element.save()
        except (KeyError, JSONDecodeError):
            return HttpResponseBadRequest()
        return JsonResponse({"id": element.id, "name": element.name}, status=201)


## Tag
@require_http_methods(['GET'])
def general_tag(request):
    """
    GET : get tag list
    """
    if request.method == 'GET':
        try:
            result = []
            for tr_elem in Tag.objects.all():
                class_elem = tr_elem.tag_class
                # tr_elem.tag_class(FK field) : TagClass Object
                result.append(
                    {
                        "id": tr_elem.id,
                        "name": tr_elem.name,
                        "color": tr_elem.color,
                        "tag_class": {
                            "id": class_elem.id,
                            "name": class_elem.name,
                            "color": class_elem.color,
                        },
                    }
                )
            return JsonResponse({"elements": result}, safe=False)
        except (KeyError, JSONDecodeError, TagClass.DoesNotExist):
            print("ERROR from general_tag")
            return HttpResponseBadRequest()
