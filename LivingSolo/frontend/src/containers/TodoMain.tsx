/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { Calendar } from '../components/Todo/Calendar';
import { CalTodoDay } from '../utils/DateTime';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { fetchTodos, selectTodo } from '../store/slices/todo';

const TodoMain = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { elements } = useSelector(selectTodo);

  const today = new Date();
  const [curDay, setCurDay] = useState<CalTodoDay>({year: today.getFullYear(), month: today.getMonth(), day: today.getDate()});

  useEffect(() => {
    dispatch(fetchTodos({
      yearMonth: curDay
    }));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dispatch, curDay.year, curDay.month]);
  
  return (
    <Wrapper>
      <InnerWrapper>
        <LeftWrapper>
          <Calendar curDay={curDay} setCurDay={setCurDay}/>
        </LeftWrapper>
        <RightWrapper>
          <div>
            <span>{curDay.year}.{curDay.month + 1}.{curDay.day}</span>
            <div>
              {curDay.day && elements[curDay.day] && elements[curDay.day].map((todo) => {
                return <div>{todo.done && '✅'}{todo.name} | {todo.deadline}</div>
              })}
            </div>
          </div>  
        </RightWrapper>
      </InnerWrapper>
    </Wrapper>
  );
};

export default TodoMain;

const Wrapper = styled.div`
  width: 100%;
  height: 100%;
  min-height: 100vh;
  background-color: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: start;
  align-items: start;
`;

const InnerWrapper = styled.div`
  width: 100%;
  height: 92%;
  display: flex;
  flex-direction: row;
  justify-content: start;
  align-items: start;
`;

const LeftWrapper = styled.div`
  width: 900px;
  height: 100%;
  padding: 30px 0px 0px 30px;

  display: flex;
  flex-direction: column;
  
  border: 1px solid black;
`;

const RightWrapper = styled.div`
  width: 100%;
  height: 100%;
  padding: 30px 30px 0px 0px;

  display: flex;
  flex-direction: column;
  border: 1px solid black;

  > div {
    width: 100%;
    height: 100%;
  }
`;