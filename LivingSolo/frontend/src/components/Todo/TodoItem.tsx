import React, { useState } from 'react';
import { styled } from 'styled-components';
import { CalTodoDay } from '../../utils/DateTime';
import { useDispatch } from 'react-redux';
import { TodoElement, toggleTodoDone } from '../../store/slices/todo';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck } from '@fortawesome/free-solid-svg-icons';
import { AppDispatch } from '../../store';
import { getContrastYIQ } from '../../styles/color';
import { TagBubbleCompact } from '../general/TagBubble';


interface TodoElementProps {
  curDay: CalTodoDay,
  setCurDay: React.Dispatch<React.SetStateAction<CalTodoDay>>,
  todo: TodoElement,
};

const TODAY_ = new Date();
const TODAY = {year: TODAY_.getFullYear(), month: TODAY_.getMonth(), day: TODAY_.getDate()};


export const TodoItem = ({ curDay, setCurDay, todo }: TodoElementProps) => {
  const dispatch = useDispatch<AppDispatch>();
  
  const doneToggle = (id: number) => {
    dispatch(toggleTodoDone({ id }));
  };

  return <TodoElementWrapper key={todo.id}>
        <TodoElementColorCircle color={todo.color} onClick={() => doneToggle(todo.id)} title={todo.category.name} ishard={todo.is_hard_deadline.toString()}>
            {todo.done && <FontAwesomeIcon icon={faCheck} fontSize={'13'} color={getContrastYIQ(todo.color)}/>}
        </TodoElementColorCircle>
        <div>
            <span>
                {todo.name}
            </span>
        </div>
        <div>
            Priority {todo.priority} 
        </div>
        <div>
            {todo.tag.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
        </div>
        <div>
            ...
        </div>
    </TodoElementWrapper>
};

const TodoElementWrapper = styled.div`
    width: 100%;
    padding: 10px;
    border-bottom: 0.5px solid green;
    
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr;
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