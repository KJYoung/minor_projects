## ReactJS, Django로 간단한 프로젝트를 구현할 때 유용한 간단한 Skeleton.
Redux-saga 베이스로 dispatch를 진행하는 버전이다.

### How to make project skeleton?
- Redux-saga 베이스로 FE-BE Interaction.
```shell
    cd backend/ && django-admin startproject backend && cd ..   
    cd frontend/ && npx create-react-app frontend --template typescript && cd ..
    # ----------------------------------------------------------------------------------
    # JOB : Remove useless parts from frontend projects...
    # ----------------------------------------------------------------------------------
    cd frontend/ && yarn add react-router-dom react-redux axios redux redux-saga styled-components styled-reset @reduxjs/toolkit && cd ..   
    # ----------------------------------------------------------------------------------
    # JOB : Setup Axios, Redux-saga...
    # ----------------------------------------------------------------------------------
    # cd backend/ && python manage.py createsuperuser && cd ..
    # ----------------------------------------------------------------------------------
    # JOB : Prepare REST API(GET, POST, PUT, DELETE protocol).
    # ----------------------------------------------------------------------------------
```
- 2023년 6월경.   