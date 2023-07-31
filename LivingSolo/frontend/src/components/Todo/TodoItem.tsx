import React, { useState } from 'react';
import { styled } from 'styled-components';
import { CalTodoDay, GetDateTimeFormat2Django } from '../../utils/DateTime';
import { useDispatch } from 'react-redux';
import { TodoEditReqType, TodoElement, deleteTodo, duplicateTodo, editTodo, toggleTodoDone } from '../../store/slices/todo';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck } from '@fortawesome/free-solid-svg-icons';
import { AppDispatch } from '../../store';
import { getContrastYIQ } from '../../styles/color';
import { TagBubbleCompact } from '../general/TagBubble';

import DatePicker from "react-datepicker";

interface TodoElementProps {
  curDay: CalTodoDay,
  setCurDay: React.Dispatch<React.SetStateAction<CalTodoDay>>,
  todo: TodoElement,
  fnMode: boolean,
  editID: number,
  setEditID: React.Dispatch<React.SetStateAction<number>>,
};

const TODAY_ = new Date();
const TODAY = {year: TODAY_.getFullYear(), month: TODAY_.getMonth(), day: TODAY_.getDate()};

export const TodoItem = ({ curDay, setCurDay, todo, fnMode, editID, setEditID }: TodoElementProps) => {
  const dispatch = useDispatch<AppDispatch>();
  const [todoDate, setTodoDate] = useState<Date>(new Date());
  const [editObj, setEditObj] = useState<TodoEditReqType>({...todo, category: todo.category.id});

  const doneToggle = (id: number, curDone: boolean) => {
    if(curDone){
        if (window.confirm('정말 완료를 취소하시겠습니까?')) {
            dispatch(toggleTodoDone({ id }));
        }
    }else{
        dispatch(toggleTodoDone({ id }));
    }
  };

  const editHandler = (id: number) => {
    if(editID === id){
        // Edit Complete.
        dispatch(editTodo({
            ...editObj,
            deadline: GetDateTimeFormat2Django(todoDate),
        }));
        setEditID(-1);
    }else{
        setEditID(id);
    }
  };

  const deleteHandler = (id: number) => {
    if (window.confirm('정말 투두를 삭제하시겠습니까?')) {
        dispatch(deleteTodo(id));
    }
  };

  return <TodoElementWrapper key={todo.id}>
        <TodoElementColorCircle color={todo.color} onClick={() => doneToggle(todo.id, todo.done)} title={todo.category.name} ishard={todo.is_hard_deadline.toString()}>
            {todo.done && <FontAwesomeIcon icon={faCheck} fontSize={'13'} color={getContrastYIQ(todo.color)}/>}
        </TodoElementColorCircle>
        <div>
            <span>
                {todo.name}{`  <${todo.priority}>`}
            </span>
        </div>
        <div>
            {todo.tag.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
        </div>
        <div>
            {editID === todo.id && <DatePicker selected={todoDate} onChange={(date: Date) => setTodoDate(date)} />}
        </div>
        <div>
            {fnMode && <>
                <button onClick={() => editHandler(todo.id)}>{editID === todo.id ? '완료' : '수정'}</button>
                <button onClick={() => dispatch(duplicateTodo(todo.id))}>복제</button>
                <button onClick={() => deleteHandler(todo.id)}>삭제</button>    
            </>}
        </div>
    </TodoElementWrapper>
};

const TodoElementWrapper = styled.div`
    width: 100%;
    padding: 10px;
    border-bottom: 0.5px solid green;
    
    display: grid;
    grid-template-columns: 1fr 10fr 5fr 2fr 4fr;
    align-items: center;

    &:last-child {
        border-bottom: none;
    };
`;

const TodoElementColorCircle = styled.div<{ color: string, ishard: string }>`
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: ${props => ((props.ishard === 'true') ? '2px solid var(--ls-red)' : 'none')};
    background-color: ${props => (props.color)};;
    
    margin-right: 10px;

    display: flex;
    justify-content: center;
    align-items: center;

    cursor: pointer;
`;