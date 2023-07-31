import React, { useEffect, useState } from 'react';
import { styled } from 'styled-components';
import { CalTodoDay, GetDateTimeFormat2Django } from '../../utils/DateTime';
import { useDispatch, useSelector } from 'react-redux';
import { TodoCategory, TodoCategoryCreateReqType, TodoCreateReqType, TodoElement, createTodo, createTodoCategory, deleteTodoCategory, selectTodo } from '../../store/slices/todo';
import { AppDispatch } from '../../store';
import { TagInputForTodo, TagInputForTodoCategory } from '../Trxn/TagInput';
import { TagElement } from '../../store/slices/tag';
import { TodoItem } from './TodoItem';
import { CondRendAnimState, toggleCondRendAnimState, condRendMounted, condRendUnmounted, onAnimEnd } from '../../utils/Rendering';
import { getRandomHex } from '../../styles/color';
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
const todoCategorySkeleton = {
  name: '',
  color: '#000000',
}

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
};

export const DailyTodo = ({ curDay, setCurDay }: DailyTodoProps) => {
  const dispatch = useDispatch<AppDispatch>();
  const { elements, categories } = useSelector(selectTodo);

  // Todo List
  const [addMode, setAddMode] = useState<CondRendAnimState>({ isMounted: false, showElem: false });
  const [fnMode, setFnMode] = useState<boolean>(false);
  const [editID, setEditID] = useState<number>(-1);
  const [categorySort, setCategorySort] = useState<boolean>(true);

  // Category List
  const [categoryPanel, setCategoryPanel] = useState<boolean>(false);
  const [categoryDelete, setCategoryDelete] = useState<boolean>(false);

  // Todo Create
  const [isPeriodic, setIsPeriodic] = useState<boolean>(false);
  const [tags, setTags] = useState<TagElement[]>([]);
  const [curTCateg, setCurTCateg] = useState<TodoCategory | null>(null);
  const [newTodo, setNewTodo] = useState<TodoCreateReqType>({...todoSkeleton, tag: tags});

  // TodoCategory Create
  const [categTags, setCategTags] = useState<TagElement[]>([]);
  const [newTodoCategory, setNewTodoCategory] = useState<TodoCategoryCreateReqType>({...todoCategorySkeleton, tag: categTags});


  const toggleCategoryPanel = () => {
    if(categoryPanel) { // T => F
        setCategoryPanel(false);
        setAddMode({ isMounted: false, showElem: false});
    }else{ // F => T
        setCategoryPanel(true);
        setAddMode({ isMounted: false, showElem: false});
    };
  };

  useEffect(() => {
    setCategoryPanel(false);
    setAddMode({ isMounted: false, showElem: false});
  }, [curDay.day]);

  return <DailyTodoWrapper>
    <DayHeaderRow className='noselect'>
        <DayH1>{curDay.year}년 {curDay.month + 1}월 {curDay.day}{curDay.day && '일'}</DayH1>
        <DayFn>
            <DayFnBtn onClick={() => toggleCategoryPanel()}>
                {categoryPanel && <span>돌아가기</span>}
                {!categoryPanel && <><span>카테고리</span><span>관리</span></>}
            </DayFnBtn>

            {categoryPanel && <>
                <DayFnBtn onClick={() => toggleCondRendAnimState(addMode, setAddMode)}>
                    {addMode.showElem ? <>
                        <span>추가</span><span>완료</span>
                    </> : <>
                        <span>카테고리</span><span>추가</span>
                    </>}
                </DayFnBtn>
                <DayFnBtn onClick={() => setCategoryDelete((cD) => !cD)}>
                    {categoryDelete ? <>
                        <span>삭제</span><span>완료</span>
                    </> : <>
                        <span>카테고리</span><span>삭제</span>
                    </>}
                </DayFnBtn>
            </>}

            {!categoryPanel && <>
                <DayFnBtn onClick={() => toggleCondRendAnimState(addMode, setAddMode)}>
                    {addMode.showElem ? <>
                        <span>추가</span><span>완료</span>
                    </> : <>
                        <span>투두</span><span>추가</span>
                    </>}
                </DayFnBtn>
                <DayFnBtn onClick={() => setCategorySort((cM) => !cM)}>
                    {categorySort && <span>중요도</span>}
                    {!categorySort && <span>카테고리</span>}
                    <span>정렬</span>
                </DayFnBtn>     
                <DayFnBtn onClick={() => setCurDay(TODAY)}>오늘로</DayFnBtn>
                <DayFnBtn onClick={() => setFnMode(fM => !fM)}>
                    {fnMode && <><span>기능</span><span>끄기</span></>}
                    {!fnMode && <><span>추가</span><span>기능</span></>}
                </DayFnBtn>
            </>}
        </DayFn>
    </DayHeaderRow>
    <DayBodyRow>
        {categoryPanel && <div>
            {addMode.showElem && <TodoAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted} onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
                <CategoryAdderRow>
                    <TodoElementColorCircle color={newTodoCategory.color} ishard={'false'} onClick={() => setNewTodoCategory((nTC) => { return {...nTC, color: getRandomHex()}})}></TodoElementColorCircle>
                    <TagInputForTodoCategory tags={categTags} setTags={setCategTags} closeHandler={() => {}}/>
                    <CategoryAdderInputWrapper>
                        <input type="text" placeholder='Category Name' value={newTodoCategory.name} onChange={(e) => setNewTodoCategory((nTC) => { return {...nTC, name: e.target.value}})}/>
                        <button onClick={() => { 
                            dispatch(createTodoCategory({...newTodoCategory, tag: categTags}));
                            setCategTags([]);
                            setNewTodoCategory({...todoCategorySkeleton, tag: categTags});
                        }}>Create</button>
                    </CategoryAdderInputWrapper>
                </CategoryAdderRow>
                
            </TodoAdderWrapper>}

            <TodoElementList style={addMode.showElem && addMode.isMounted ? { transform: "translateY(55px)" } : { transform: "translateY(0px)" }}>
                {categories.map((categ) => <TodoCategoryItem key={categ.id}>
                        <TodoElementColorCircle color={categ.color} ishard='false' />
                        <span>{categ.name}</span>
                        <div>
                            {categ.tag.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
                        </div>
                        {categoryDelete && <div className='clickable' onClick={() => {
                            if (window.confirm('정말 카테고리를 삭제하시겠습니까?')) {
                                dispatch(deleteTodoCategory(categ.id));
                            }}}>
                            삭제
                        </div>}
                </TodoCategoryItem>)}
            </TodoElementList>
        </div>}
        {!categoryPanel && <>
            {addMode.showElem && <TodoAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted} onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
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
                            console.log('categ', categ);
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
                        <button onClick={() => { 
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
            </TodoAdderWrapper>}
            {/* Important! `addMode.showElem && addMode.isMounted` <== is required for the smooth transition! Not only one of them, But both! */}
            <TodoElementList style={addMode.showElem && addMode.isMounted ? { transform: "translateY(125px)" } : { transform: "translateY(0px)" }}>
                {categorySort && <>
                    {curDay.day && elements[curDay.day] && categoricalSlicer(elements[curDay.day])
                        .map((categoryElement) => {
                            return <TodoCategoryWrapper key={categoryElement.id}>
                                <TodoCategoryHeader>
                                    <TodoElementColorCircle color={categoryElement.color} ishard='false' />
                                    <span>{categoryElement.name}</span>
                                </TodoCategoryHeader>
                                <TodoCategoryBody>
                                {categoryElement.todos // For Read Only Array Sort, We have to copy that.
                                    .sort((a, b) => b.priority - a.priority) // Descending Order! High Priority means Important Job.
                                    .map((todo) => {
                                        return <TodoItem key={todo.id} todo={todo} curDay={curDay} setCurDay={setCurDay}
                                                        fnMode={fnMode} editID={editID} setEditID={setEditID}/>
                                })}
                                </TodoCategoryBody>
                            </TodoCategoryWrapper>
                        })}
                </>}
                {!categorySort && <>
                    {curDay.day && elements[curDay.day] && [...elements[curDay.day]] // For Read Only Array Sort, We have to copy that.
                    .sort((a, b) => b.priority - a.priority) // Descending Order! High Priority means Important Job.
                    .map((todo) => {
                        return <TodoItem key={todo.id} todo={todo} curDay={curDay} setCurDay={setCurDay}
                                        fnMode={fnMode}  editID={editID} setEditID={setEditID}/>
                })}
                </>}
            </TodoElementList>
        </>}
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

    position: relative;
`;

const TodoAdderWrapper = styled.div`
    width: 100%;
    display: flex;
    flex-direction: column;

    margin-bottom: 10px;
`;
const CategoryAdderRow = styled.div`
    display: grid;
    grid-template-columns: 1fr 5fr 13fr;
    align-items: center;

    padding: 4px;
    padding-bottom: 10px;
    border-bottom: 1.5px solid gray;
`;
const CategoryAdderInputWrapper = styled.div`
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

const TodoCategoryItem = styled.div`
    width  : 100%;

    display: grid;
    grid-template-columns: 1fr 12fr 5fr 2fr;
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
const TodoElementList = styled.div`
    width: 100%;
    position: absolute;
    top: 0px;

    transition-property: all;
    transition-duration: 250ms;
    transition-delay: 0s;
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