import React from 'react';
import { styled } from 'styled-components';
import { CalTodoDay } from '../../utils/DateTime';
import { useDispatch, useSelector } from 'react-redux';
import { selectTodo, toggleTodoDone } from '../../store/slices/todo';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck } from '@fortawesome/free-solid-svg-icons';
import { AppDispatch } from '../../store';
import { getContrastYIQ } from '../../styles/color';


interface DailyTodoProps {
  curDay: CalTodoDay,
  setCurDay: React.Dispatch<React.SetStateAction<CalTodoDay>>,
};

const TODAY_ = new Date();
const TODAY = {year: TODAY_.getFullYear(), month: TODAY_.getMonth(), day: TODAY_.getDate()};

export const DailyTodo = ({ curDay, setCurDay }: DailyTodoProps) => {
  const dispatch = useDispatch<AppDispatch>();
  const { elements } = useSelector(selectTodo);

  const doneToggle = (id: number) => {
    dispatch(toggleTodoDone({ id }));
  };

  return <DailyTodoWrapper>
    <DayHeaderRow>
        <DayH1>{curDay.year}년 {curDay.month + 1}월 {curDay.day}{curDay.day && '일'}</DayH1>
        <DayFn>
            <DayFnBtn >더보기</DayFnBtn>
            <DayFnBtn onClick={() => setCurDay(TODAY)}>오늘로</DayFnBtn>
        </DayFn>
    </DayHeaderRow>
    <DayBodyRow>
        {curDay.day && elements[curDay.day] && elements[curDay.day]
            .map((todo) => {
                return <TodoElementWrapper>
                    <TodoElementColorCircle color={todo.color} onClick={() => doneToggle(todo.id)}>
                        {todo.done && <FontAwesomeIcon icon={faCheck} fontSize={'13'} color={getContrastYIQ(todo.color)}/>}
                    </TodoElementColorCircle>
                    {todo.name} | {todo.is_hard_deadline ? 'HARD' : 'SOFT'} | {todo.priority} | {todo.period} | {todo.category.name} | {todo.tag.map((t) => t.name)}
                </TodoElementWrapper>
        })}
    </DayBodyRow>
</DailyTodoWrapper>
};

const DailyTodoWrapper = styled.div`
  width: 800px;
  min-height: 800px;
  max-height: 800px;
  display: grid;
  grid-template-rows: 1fr 9fr;
  
  margin-left: 10px;
`;

const DayHeaderRow = styled.div`
  width: 100%;
  margin-top: 30px;
  border-bottom: 0.5px solid gray;

  display: flex;
`;
const DayH1 = styled.span`
  width: 100%;
  font-size: 40px;
  color: var(--ls-gray);
`;
const DayFn = styled.div`
  width: 100px;
  align-self: flex-end;
  display: flex;
  flex-direction: column;
  align-items: center;
`;
const DayFnBtn = styled.button`
    background-color: transparent;
    border: none;
    width: 100%;
    font-size: 15px;
    color: var(--ls-gray_google2);
    cursor: pointer;
    &:hover {
        color: var(--ls-blue);
    }
    &:not(:first-child) {
        border-top: 1px solid var(--ls-gray);
        padding-top: 5px;
    }
    margin-bottom: 3px;
    margin-left: 5px;
`;

const DayBodyRow = styled.div`
    width: 100%;
    margin-top: 20px;
`;

const TodoElementWrapper = styled.div`
    width: 100%;
    padding: 10px;
    border-bottom: 0.5px solid green;
    
    display: flex;
    align-items: center;

    &:last-child {
        border-bottom: none;
    };
`;
const TodoElementColorCircle = styled.div<{ color: string }>`
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background-color: ${props => (props.color)};;
    
    margin-right: 10px;

    display: flex;
    justify-content: center;
    align-items: center;

    cursor: pointer;
`;