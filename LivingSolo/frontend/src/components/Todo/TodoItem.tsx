import React from 'react';
import { styled } from 'styled-components';
import { useDispatch } from 'react-redux';
import { TodoElement, deleteTodo, dupAgainTodo, duplicateTodo, toggleTodoDone } from '../../store/slices/todo';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck } from '@fortawesome/free-solid-svg-icons';
import { AppDispatch } from '../../store';
import { getContrastYIQ } from '../../styles/color';
import { TagBubbleCompact } from '../general/TagBubble';
import { A_EQUAL_B_CalTodoDay, A_LESS_THAN_B_CalTodoDay, GetDateTimeFormat2Django, TODAY, TODAY_, TOMORROW_, calTodoDayConst } from '../../utils/DateTime';

interface TodoElementProps {
  todo: TodoElement,
  fnMode: boolean,
  editID: number,
  setEditID: React.Dispatch<React.SetStateAction<number>>,
  setEditMode: () => void, 
};

export const TodoItem = ({ todo, fnMode, editID, setEditID, setEditMode }: TodoElementProps) => {
  const dispatch = useDispatch<AppDispatch>();

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
    setEditID(id);
    setEditMode();
  };

  const deleteHandler = (id: number) => {
    if (window.confirm('정말 투두를 삭제하시겠습니까?')) {
        dispatch(deleteTodo(id));
    }
  };

  const dupAgainHandler = (id: number, dateObj: Date) => {
    dispatch(dupAgainTodo({
        todoID: id,
        date: GetDateTimeFormat2Django(dateObj),
    }));
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
            {fnMode && <>
                {A_EQUAL_B_CalTodoDay(calTodoDayConst(todo.deadline), TODAY) && <button onClick={() => dupAgainHandler(todo.id, TOMORROW_)}>내일또하기</button>}
                {A_LESS_THAN_B_CalTodoDay(calTodoDayConst(todo.deadline), TODAY) && <button onClick={() => dupAgainHandler(todo.id, TODAY_)}>오늘또하기</button>}
                <button onClick={() => editHandler(todo.id)}>수정</button>
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
    grid-template-columns: 1fr 10fr 5fr 4fr;
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