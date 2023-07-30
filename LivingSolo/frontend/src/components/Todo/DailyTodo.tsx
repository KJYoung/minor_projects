import React, { useState } from 'react';
import { styled } from 'styled-components';
import { CalTodoDay, GetDateTimeFormat2Django } from '../../utils/DateTime';
import { useDispatch, useSelector } from 'react-redux';
import { TodoCategory, TodoCreateReqType, TodoElement, createTodo, selectTodo } from '../../store/slices/todo';
import { AppDispatch } from '../../store';
import { TagInputForTodo } from '../Trxn/TagInput';
import { TagElement } from '../../store/slices/tag';
import { TodoItem } from './TodoItem';


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

interface CategoricalTodos {
    id: number,
    name: string,
    color: string,
    todos: TodoElement[]
};
const categoricalSlicer = (todoItems : TodoElement[]) : CategoricalTodos[] => {
    const result: CategoricalTodos[] = [];

    const alreadyCategory = (id: number, already: CategoricalTodos[]) => already.findIndex((res) => res.id === id);

    todoItems.forEach((todo) => {
        const categoryIndex = alreadyCategory(todo.category.id, result);
        if(categoryIndex !== -1){
            result[categoryIndex].todos.push(todo);
        }else{
            result.push({
                ...todo.category,
                todos: [todo]
            });
        };
    });
    return result;
}

export const DailyTodo = ({ curDay, setCurDay }: DailyTodoProps) => {
  const dispatch = useDispatch<AppDispatch>();
  const { elements, categories } = useSelector(selectTodo);

  const [addMode, setAddMode] = useState<boolean>(false);
  const [categoryMode, setCategoryMode] = useState<boolean>(true);
  const [isPeriodic, setIsPeriodic] = useState<boolean>(false);
  const [tags, setTags] = useState<TagElement[]>([]);
  const [curTCateg, setCurTCateg] = useState<TodoCategory | null>(null);
  const [newTodo, setNewTodo] = useState<TodoCreateReqType>({...todoSkeleton, tag: tags});

  return <DailyTodoWrapper>
    <DayHeaderRow>
        <DayH1>{curDay.year}년 {curDay.month + 1}월 {curDay.day}{curDay.day && '일'}</DayH1>
        <DayFn>
            <DayFnBtn onClick={() => setAddMode((aM) => !aM)}>
                <span>투두</span><span>추가</span>
            </DayFnBtn>
            <DayFnBtn onClick={() => setCategoryMode((cM) => !cM)}>
                {categoryMode && <span>중요도</span>}
                {!categoryMode && <span>카테고리</span>}
                <span>정렬</span>
            </DayFnBtn>
            <DayFnBtn>
                <span>카테고리</span><span>관리</span>
            </DayFnBtn>
            <DayFnBtn onClick={() => setCurDay(TODAY)}>오늘로</DayFnBtn>
        </DayFn>
    </DayHeaderRow>
    <DayBodyRow>
        {addMode && <TodoAdderWrapper>
            <TodoAdder1stRow>
                <TodoAdderAddInputs>
                    <label htmlFor="prioritySelect">중요도</label>
                    <select id="prioritySelect" value={newTodo.priority} onChange={(e) => { setNewTodo((nT) => { return {...nT, priority: parseInt(e.target.value)}}); }}>
                        <option disabled value={0}>- 중요도 -</option>
                        {Array(10).fill(null).map((_, index) => <option value={index + 1} key={index + 1}>{index + 1}</option>)}
                    </select>
                </TodoAdderAddInputs>
                <TodoAdderAddInputs>
                    <TodoPeriodicLabel>
                        <label htmlFor="periodInput">주기</label>
                        <input  type="checkbox" id="isPeriodic" checked={isPeriodic}
                            onChange={(e) => {
                                setIsPeriodic((ip) => !ip);
                                setNewTodo((nT) => { return {...nT, period: 0}});
                            }} />
                    </TodoPeriodicLabel>
                    <input id="periodInput" type="number" disabled={!isPeriodic} value={newTodo.period} onChange={(e) => setNewTodo((nT) => { return {...nT, period: parseInt(e.target.value)}})}/>
                </TodoAdderAddInputs>
                <TodoAdderAddInputs>
                    <label htmlFor="ishardDeadline">엄격성</label>
                    <input  type="checkbox" id="ishardDeadline" checked={newTodo.is_hard_deadline}
                        onChange={(e) => setNewTodo((nT) => { return {...nT, is_hard_deadline: !nT.is_hard_deadline}})} />
                </TodoAdderAddInputs>
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
                <TagInputForTodo tags={tags} setTags={setTags} closeHandler={() => {}}/>
            </TodoAdder1stRow>
            <TodoAdder2ndRow>
                <TodoElementColorCircle color={curTCateg ? curTCateg.color : 'gray'} ishard={newTodo.is_hard_deadline.toString()}></TodoElementColorCircle>
                <TodoAdder2nfRowInputWrapper>
                    <input type="text" placeholder='Todo Name' value={newTodo.name} onChange={(e) => setNewTodo((nT) => { return {...nT, name: e.target.value}})}/>
                    <button onClick={() => { curDay.day && dispatch(createTodo({
                        ...newTodo,
                        tag: tags,
                        deadline: GetDateTimeFormat2Django(new Date(curDay.year, curDay.month, curDay.day)),
                    })) }}>Create</button>
                </TodoAdder2nfRowInputWrapper>
            </TodoAdder2ndRow>
        </TodoAdderWrapper>}
        <TodoElementList>
            {categoryMode && <>
                {curDay.day && elements[curDay.day] && categoricalSlicer(elements[curDay.day])
                    .map((categoryElement) => {
                        return <TodoCategoryWrapper>
                            <TodoCategoryHeader>
                                <TodoElementColorCircle color={categoryElement.color} ishard='false' />
                                <span>{categoryElement.name}</span>
                            </TodoCategoryHeader>
                            <TodoCategoryBody>
                            {categoryElement.todos // For Read Only Array Sort, We have to copy that.
                                .sort((a, b) => b.priority - a.priority) // Descending Order! High Priority means Important Job.
                                .map((todo) => {
                                    return <TodoItem todo={todo} curDay={curDay} setCurDay={setCurDay} />
                            })}
                            </TodoCategoryBody>
                        </TodoCategoryWrapper>
                    })}
            </>}
            {!categoryMode && <>
                {curDay.day && elements[curDay.day] && [...elements[curDay.day]] // For Read Only Array Sort, We have to copy that.
                .sort((a, b) => b.priority - a.priority) // Descending Order! High Priority means Important Job.
                .map((todo) => {
                    return <TodoItem todo={todo} curDay={curDay} setCurDay={setCurDay} />
            })}
            </>}
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
  width: 500px;
  font-size: 40px;
  color: var(--ls-gray);
`;
const DayFn = styled.div`
  width: 100%;
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
    display: flex;
    flex-direction: column;

    margin-bottom: 10px;
`;
const TodoAdder1stRow = styled.div`
    width: 100%;
    display: grid;
    grid-gap: 15px;
    grid-template-columns: 1fr 1fr 1fr 1fr 3fr;
    
    margin-bottom: 10px;
`;
const TodoAdderAddInputs = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    label {
        color: var(--ls-gray_google2);
        margin-bottom: 2px;
    }
`;
const TodoPeriodicLabel = styled.div`
    margin-bottom: 2px;

    input {
        margin-left: 10px;
    }
`;
const TodoAdder2ndRow = styled.div`
    width: 100%;
    padding: 10px 10px 15px 10px;
    border-bottom: 1.5px solid gray;
    margin-bottom: 10px;
    
    display: flex;
    align-items: center;
`;
const TodoAdder2nfRowInputWrapper = styled.div`
    width  : 100%;
    display: flex;
    justify-content: space-between;

    input {
        width: 100%;
        padding: 10px;
        margin-right: 20px;
    }
    button {
        padding: 10px;
    }
`;
const TodoElementList = styled.div`
    width: 100%;
`;

const TodoCategoryWrapper = styled.div`
    width  : 100%;
`;
const TodoCategoryHeader = styled.div`
    width  : 100%;

    display: flex;
    align-items: center;
    
    padding: 4px;
    margin-top: 10px;
    border-bottom: 1px solid var(--ls-gray);

    > span {
        font-size: 24px;
        font-weight: 300;
        /* color: var(--ls-gray); */
    }
`;
const TodoCategoryBody = styled.div`
    width  : 100%;
    margin-left: 10px;
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