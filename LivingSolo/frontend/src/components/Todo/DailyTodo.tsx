import React, { useState } from 'react';
import { styled } from 'styled-components';
import { CalTodoDay, GetDateTimeFormat2Django } from '../../utils/DateTime';
import { useDispatch, useSelector } from 'react-redux';
import { TodoCategory, TodoCreateReqType, createTodo, selectTodo, toggleTodoDone } from '../../store/slices/todo';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck } from '@fortawesome/free-solid-svg-icons';
import { AppDispatch } from '../../store';
import { getContrastYIQ } from '../../styles/color';
import { TagInputForGridHeader } from '../Trxn/TagInput';
import { TagElement } from '../../store/slices/tag';
import { TagBubbleCompact } from '../general/TagBubble';


interface DailyTodoProps {
  curDay: CalTodoDay,
  setCurDay: React.Dispatch<React.SetStateAction<CalTodoDay>>,
};

const TODAY_ = new Date();
const TODAY = {year: TODAY_.getFullYear(), month: TODAY_.getMonth(), day: TODAY_.getDate()};

const DEFAULT_OPTION = '$NONE$';
const todoSkeleton = {
  name: '',
  category: DEFAULT_OPTION,
  priority: 0,
  deadline: '',
  is_hard_deadline: false,
  period: 0,
};

export const DailyTodo = ({ curDay, setCurDay }: DailyTodoProps) => {
  const dispatch = useDispatch<AppDispatch>();
  const { elements, categories } = useSelector(selectTodo);

  const [addMode, setAddMode] = useState<boolean>(false);
  const [tags, setTags] = useState<TagElement[]>([]);
  const [curTCateg, setCurTCateg] = useState<TodoCategory | null>(null);
  const [newTodo, setNewTodo] = useState<TodoCreateReqType>({...todoSkeleton, tag: tags});
  
  const doneToggle = (id: number) => {
    dispatch(toggleTodoDone({ id }));
  };

  return <DailyTodoWrapper>
    <DayHeaderRow>
        <DayH1>{curDay.year}년 {curDay.month + 1}월 {curDay.day}{curDay.day && '일'}</DayH1>
        <DayFn>
            <DayFnBtn onClick={() => setAddMode((aM) => !aM)}>
                <span>투두</span>
                <span>추가</span>
            </DayFnBtn>
            <DayFnBtn>
                <span>카테고리</span>
                <span>관리</span>
            </DayFnBtn>
            <DayFnBtn onClick={() => setCurDay(TODAY)}>오늘로</DayFnBtn>
        </DayFn>
    </DayHeaderRow>
    <DayBodyRow>
        {addMode && <TodoAdderWrapper>
            <input type="text" placeholder='Todo Name' value={newTodo.name} onChange={(e) => setNewTodo((nT) => { return {...nT, name: e.target.value}})}/>
            <input type="number" value={newTodo.period} onChange={(e) => setNewTodo((nT) => { return {...nT, period: parseInt(e.target.value)}})}/>
            <input type="number" value={newTodo.priority} onChange={(e) => setNewTodo((nT) => { return {...nT, priority: parseInt(e.target.value)}})}/>
            <label htmlFor="isHardDeadline">엄격한?</label>
            <input  type="checkbox" id="isHardDeadline" checked={newTodo.is_hard_deadline}
                    onChange={(e) => setNewTodo((nT) => { return {...nT, is_hard_deadline: !nT.is_hard_deadline}})} />
            <select value={newTodo.category} onChange={(e) => {
                setNewTodo((nT) => { return {...nT, category: e.target.value}});
                const categ = categories.find((c) => c.id === parseInt(e.target.value));
                categ && setCurTCateg(categ);
            }}>
                  <option disabled value={DEFAULT_OPTION}>
                    - 투두 카테고리 -
                  </option>
                  {categories.map(categ => {
                      return (
                          <option value={categ.id} key={categ.id}>
                          {categ.name}
                        </option>
                      );
                    })}
            </select>
            <TagInputForGridHeader tags={tags} setTags={setTags} closeHandler={() => {}}/>
            {curTCateg && <TodoElementColorCircle color={curTCateg.color}></TodoElementColorCircle>}
            <button onClick={() => { curDay.day && dispatch(createTodo({
                ...newTodo,
                tag: tags,
                deadline: GetDateTimeFormat2Django(new Date(curDay.year, curDay.month, curDay.day)),
            })) }}>Create</button>
        </TodoAdderWrapper>}
        <TodoElementList>
            {curDay.day && elements[curDay.day] && elements[curDay.day]
                .map((todo) => {
                    return <TodoElementWrapper key={todo.id}>
                        <TodoElementColorCircle color={todo.color} onClick={() => doneToggle(todo.id)} title={todo.category.name}>
                            {todo.done && <FontAwesomeIcon icon={faCheck} fontSize={'13'} color={getContrastYIQ(todo.color)}/>}
                        </TodoElementColorCircle>
                        {todo.name} | {todo.is_hard_deadline ? 'HARD' : 'SOFT'} | {todo.priority} | {todo.period} | {todo.category.name} | 
                            {todo.tag.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
                    </TodoElementWrapper>
            })}
        </TodoElementList>
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
  width: 300px;
  height: 100%;
  align-self: flex-end;
  display: flex;
  flex-direction: row;
  align-items: center;
`;
const DayFnBtn = styled.div`
    width: 100%;
    height: 100%;
    font-size: 15px;
    color: var(--ls-gray_google2);
    cursor: pointer;
    &:hover {
        color: var(--ls-blue);
    }
    &:not(:first-child) {
        border-left: 1px solid var(--ls-gray);
    }
    > span:not(:first-child) {
        margin-top: 3px;
    }
    margin-bottom: 3px;
    margin-left: 5px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
`;

const DayBodyRow = styled.div`
    width: 100%;
    margin-top: 20px;
`;

const TodoAdderWrapper = styled.div`
    width: 100%;
`;
const TodoElementList = styled.div`
    width: 100%;
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