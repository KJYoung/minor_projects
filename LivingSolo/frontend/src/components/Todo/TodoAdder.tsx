import { styled } from "styled-components";
import { CondRendAnimState, condRendMounted, condRendUnmounted, onAnimEnd } from "../../utils/Rendering";
import { useState } from "react";
import { TagElement } from "../../store/slices/tag";
import { TodoCategory, TodoCreateReqType, TodoEditReqType, TodoElement, createTodo, editTodo as editTodoDispatch, selectTodo } from "../../store/slices/todo";
import { useDispatch, useSelector } from "react-redux";
import { TagInputForTodo } from "../Tag/TagInput";
import { AppDispatch } from "../../store";
import { CalTodoDay, GetDateObjFromDjango, GetDateTimeFormat2Django } from "../../utils/DateTime";

import DatePicker from "react-datepicker";
import { DEFAULT_OPTION } from "../../utils/Constants";

interface TodoAdderProps {
    addMode: CondRendAnimState,
    setAddMode: React.Dispatch<React.SetStateAction<CondRendAnimState>>,
    curDay: CalTodoDay,
};

interface TodoEditorProps extends TodoAdderProps {
    editObj: TodoElement,
    editCompleteHandler: () => void,
};

const todoSkeleton = {
    name: '',
    category: DEFAULT_OPTION,
    priority: 0,
    deadline: '',
    is_hard_deadline: false,
    period: 0,
};

export const TodoAdder = ({ addMode, setAddMode, curDay } : TodoAdderProps) => {
    const dispatch = useDispatch<AppDispatch>();
    const { categories } = useSelector(selectTodo);

    // Todo List - Create
    const [isPeriodic, setIsPeriodic] = useState<boolean>(false);
    const [tags, setTags] = useState<TagElement[]>([]);
    const [curTCateg, setCurTCateg] = useState<TodoCategory | null>(null);
    const [newTodo, setNewTodo] = useState<TodoCreateReqType>({...todoSkeleton, tag: tags});

    return <TodoAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted}
                            onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
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
            if(categ){
                setCurTCateg(categ);
                setTags(categ.tag);
            };
        }}>
            <option disabled value={DEFAULT_OPTION}>
                - 카테고리 -
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
        <TodoAdder2ndRowInputWrapper>
            <input type="text" placeholder='Todo Name' value={newTodo.name} onChange={(e) => setNewTodo((nT) => { return {...nT, name: e.target.value}})}/>
            <button disabled={curTCateg === null || newTodo.name === ''} onClick={() => { 
                if(curDay.day){
                    dispatch(createTodo({
                        ...newTodo,
                        tag: tags,
                        deadline: GetDateTimeFormat2Django(new Date(curDay.year, curDay.month, curDay.day)),
                    }));
                    setTags([]);
                    setNewTodo({...todoSkeleton, tag: tags});
                }else{
                    // ERROR
                }
            }}>Create</button>
        </TodoAdder2ndRowInputWrapper>
    </TodoAdder2ndRow>
</TodoAdderWrapper>
};

export const TodoEditor = ({ addMode, setAddMode, curDay, editObj, editCompleteHandler } : TodoEditorProps) => {
    const dispatch = useDispatch<AppDispatch>();
    const { categories } = useSelector(selectTodo);

    // Todo List - Edit
    const [isPeriodic, setIsPeriodic] = useState<boolean>(editObj.period > 0);
    const [tags, setTags] = useState<TagElement[]>(editObj.tag);
    const [curTCateg, setCurTCateg] = useState<TodoCategory | null>(editObj.category);
    const [editTodo, setEditTodo] = useState<TodoEditReqType>({ ...editObj, category: editObj.category.id });
    const [todoDate, setTodoDate] = useState<Date>(GetDateObjFromDjango(editObj.deadline));

    return <TodoAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted}
                            onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
    <TodoEditor1stRow>
        <TodoAdderAddInputs>
            <label htmlFor="prioritySelect">날짜</label>
            <DatePicker selected={todoDate} onChange={(date: Date) => setTodoDate(date)} />
        </TodoAdderAddInputs>
        <TodoAdderAddInputs>
            <label htmlFor="prioritySelect">중요도</label>
            <select id="prioritySelect" value={editTodo.priority} onChange={(e) => { setEditTodo((eT) => { return {...eT, priority: parseInt(e.target.value)}}); }}>
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
                        setEditTodo((eT) => { return {...eT, period: 0}});
                    }} />
            </TodoPeriodicLabel>
            <input id="periodInput" type="number" disabled={!isPeriodic} value={editTodo.period} onChange={(e) => setEditTodo((eT) => { return {...eT, period: parseInt(e.target.value)}})}/>
        </TodoAdderAddInputs>
        <TodoAdderAddInputs>
            <label htmlFor="ishardDeadline">엄격성</label>
            <input  type="checkbox" id="ishardDeadline" checked={editTodo.is_hard_deadline}
                onChange={(e) => setEditTodo((eT) => { return {...eT, is_hard_deadline: !eT.is_hard_deadline}})} />
        </TodoAdderAddInputs>
        <select value={editTodo.category} onChange={(e) => {
            setEditTodo((eT) => { return {...eT, category: parseInt(e.target.value)}});
            const categ = categories.find((c) => c.id === parseInt(e.target.value));
            if(categ){
                setCurTCateg(categ);
                setTags(categ.tag);
            };
        }}>
            <option disabled value={DEFAULT_OPTION}>
                - 카테고리 -
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
    </TodoEditor1stRow>
    <TodoAdder2ndRow>
        <TodoElementColorCircle color={curTCateg ? curTCateg.color : 'gray'} ishard={editTodo.is_hard_deadline.toString()}></TodoElementColorCircle>
        <TodoAdder2ndRowInputWrapper>
            <input type="text" placeholder='Todo Name' value={editTodo.name} onChange={(e) => setEditTodo((eT) => { return {...eT, name: e.target.value}})}/>
            <button disabled={curTCateg === null || editTodo.name === ''} onClick={() => { 
                if(curDay.day){
                    dispatch(editTodoDispatch({
                        ...editTodo,
                        tag: tags,
                        deadline: GetDateTimeFormat2Django(todoDate),
                    }));
                    editCompleteHandler();
                }else{
                    // ERROR
                }
            }}>Edit</button>
            <button onClick={() => editCompleteHandler()}>
                Cancel
            </button>
        </TodoAdder2ndRowInputWrapper>
    </TodoAdder2ndRow>
</TodoAdderWrapper>
};

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
const TodoEditor1stRow = styled.div`
    width: 100%;
    display: grid;
    grid-gap: 15px;
    grid-template-columns: 3fr 1fr 1fr 1fr 3fr 5fr;
    
    margin-bottom: 10px;

    input {
        max-width: 80px; // For Amount Input
        text-align: center;
    }
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
const TodoAdder2ndRowInputWrapper = styled.div`
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