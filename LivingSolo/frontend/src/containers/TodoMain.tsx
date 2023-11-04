/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { Calendar } from '../components/Todo/Calendar';
import { CalTodoDay } from '../utils/DateTime';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { fetchTodoCategory, fetchTodos, selectTodo } from '../store/slices/todo';
import { DailyTodo } from '../components/Todo/DailyTodo';
import { ERRORSTATE } from '../store/slices/core';
import { fetchTagPresets, fetchTags, fetchTagsIndex } from '../store/slices/tag';
import { useSearchParams } from 'react-router-dom';

const TodoMain = () => {
  const dispatch = useDispatch<AppDispatch>();
  const [params] = useSearchParams();

  const today = new Date();
  const [curDay, setCurDay] = useState<CalTodoDay>({year: today.getFullYear(), month: today.getMonth(), day: today.getDate()});
  const { errorState } = useSelector(selectTodo);

  useEffect(() => {
    const year = params.get('year');
    const month = params.get('month');
    const day = params.get('day');
    if(year && month && day){
      setCurDay({ year: Number(year), month: Number(month) - 1, day: Number(day)});
    };
  }, [params]);

  useEffect(() => {
    dispatch(fetchTodos({
      yearMonth: curDay
    }));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dispatch, curDay.year, curDay.month]);
  
  useEffect(() => {
    if(errorState === ERRORSTATE.SUCCESS){
      dispatch(fetchTodos({ yearMonth: curDay }));
      dispatch(fetchTodoCategory());
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [errorState]);

  useEffect(() => {
    dispatch(fetchTodoCategory());
    dispatch(fetchTags());
    dispatch(fetchTagsIndex());
    dispatch(fetchTagPresets());
  }, [dispatch]);

  return (
    <Wrapper>
      <InnerWrapper>
        <LeftWrapper>
          <Calendar curDay={curDay} setCurDay={setCurDay}/>
        </LeftWrapper>
        <RightWrapper>
          <DailyTodo curDay={curDay} setCurDay={setCurDay}/>
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
`;

const RightWrapper = styled.div`
  width: 100%;
  height: 100%;
  padding: 30px 30px 0px 15px;

  display: flex;
  flex-direction: column;

  > div {
    width: 100%;
    height: 100%;
  }
`;