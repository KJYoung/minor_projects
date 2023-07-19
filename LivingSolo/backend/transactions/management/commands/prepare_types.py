from django.core.management import BaseCommand
from transactions.models import TransactionTypeClass, TransactionType

colors = ['#f4d284', '#f9b6a2', '#f9a2b6', '#a2cff9', '#9fd6cd', '#a9f9a2', '#d3b7d8', '#d3b7d8']


class Command(BaseCommand):
    help = "This command prepares some type classes & basic types."

    def handle(self, *args, **options):
        class_list = ["음식", "여가", "교통"]

        # 무산소 태그 프리셋
        class_sublist = {}
        class_sublist["음식"] = ["아침", "점심", "저녁", "식료품", "간식", "카페"]
        class_sublist["생활"] = ["쇼핑"]
        class_sublist["여가"] = ["여행", "모임", "게임", "운동", "도서", "영화"]
        class_sublist["교통"] = ["주유", "통행료", "대중교통", "택시"]
        class_sublist["구독"] = [
            "유튜브",
        ]
        class_sublist["관계"] = ["혼자", "과기", "팀원", "가족", "친구"]

        tag_preset = [
            class_sublist["음식"],  # "음식"
            class_sublist["여가"],  # "여가"
            class_sublist["교통"],  # "교통"
            class_sublist["구독"],  # "구독"
            class_sublist["관계"],  # "관계"
        ]

        class_ind = 0
        for class_name, tag_names in zip(class_list, tag_preset):
            type_class = TransactionTypeClass.objects.create(
                name=class_name,
                color=colors[class_ind],
            )
            class_ind += 1

            for tag_name in tag_names:
                TransactionType.objects.create(
                    name=tag_name, type_class=type_class, color=colors[class_ind]
                )

        self.stdout.write(
            self.style.SUCCESS("Type classes & Type presets are prepared automatically.")
        )
