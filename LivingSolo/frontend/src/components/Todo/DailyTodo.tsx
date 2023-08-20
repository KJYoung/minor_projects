import React, { useEffect, useState } from 'react';
import { styled } from 'styled-components';
import { CalTodoDay, DayDiffCalTodoDay, GetDjangoDateByCalTodoDay, TODAY, TOMORROW } from '../../utils/DateTime';
import { useDispatch, useSelector } from 'react-redux';
import { TodoElement, deleteTodoCategory, postponeTodo, selectTodo } from '../../store/slices/todo';
import { AppDispatch } from '../../store';
import { TodoItem } from './TodoItem';
import { CondRendAnimState, toggleCondRendAnimState, defaultCondRendAnimState } from '../../utils/Rendering';
import { TagBubbleCompact } from '../general/TagBubble';
import { TodoAdder, TodoEditor } from './TodoAdder';
import { CategoryAdder, CategoryEditor } from './CategoryAdder';
import { notificationDefault } from '../../utils/sendNoti';

enum TodoListMode {
    TodoGeneral, CategoryGeneral, TodoFunctional,
};

interface DailyTodoProps {
  curDay: CalTodoDay,
  setCurDay: React.Dispatch<React.SetStateAction<CalTodoDay>>,
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
};

interface DailyTodoHeaderProps {
    headerMode: TodoListMode,
    setHeaderMode: (tLM : TodoListMode) => void,
    addMode: CondRendAnimState,
    setAddMode: React.Dispatch<React.SetStateAction<CondRendAnimState>>,
    curDay: CalTodoDay,
    setCurDay: React.Dispatch<React.SetStateAction<CalTodoDay>>,
    categoryFn: CategoryFnMode,
    setCategoryFn: React.Dispatch<React.SetStateAction<CategoryFnMode>>,
    categorySort: boolean,
    setCategorySort: React.Dispatch<React.SetStateAction<boolean>>,
    setEditID: React.Dispatch<React.SetStateAction<number>>,
};

enum CategoryFnMode {
    LIST, ADD, EDIT, DELETE
};

const DailyTodoHeader = ({ 
                            headerMode, setHeaderMode, addMode, setAddMode, curDay, setCurDay,
                            categoryFn, setCategoryFn, categorySort, setCategorySort, setEditID,
                        }: DailyTodoHeaderProps) => 
{
    const dispatch = useDispatch<AppDispatch>();
    const { categories } = useSelector(selectTodo);

    const toggleCategoryPanel = () => {
        setHeaderMode(headerMode === TodoListMode.CategoryGeneral ? TodoListMode.TodoGeneral : TodoListMode.CategoryGeneral);
    };

    const categoryAddToggleHandler = () => {
        if(categoryFn === CategoryFnMode.LIST || categoryFn === CategoryFnMode.DELETE){
            toggleCondRendAnimState(addMode, setAddMode); // ON
            setCategoryFn(CategoryFnMode.ADD);
        }else if(categoryFn === CategoryFnMode.ADD){
            toggleCondRendAnimState(addMode, setAddMode); // OFF
            setCategoryFn(CategoryFnMode.LIST);
        }else{
            notificationDefault('Category', 'EDIT 일 때는 ADD로 전환할 수 없어요.');
        }
    };

    const categoryEditToggleHandler = () => {
        if(categoryFn === CategoryFnMode.LIST || categoryFn === CategoryFnMode.DELETE){
            toggleCondRendAnimState(addMode, setAddMode);
            setCategoryFn(CategoryFnMode.EDIT);
            setEditID(categories[0].id);
        }else if(categoryFn === CategoryFnMode.EDIT){
            toggleCondRendAnimState(addMode, setAddMode);
            setCategoryFn(CategoryFnMode.LIST);
        }else{
            notificationDefault('Category', 'ADD 일 때는 EDIT로 전환할 수 없어요.');
        }
    };

    const categoryDeleteToggleHandler = () => {
        if(categoryFn === CategoryFnMode.ADD || categoryFn === CategoryFnMode.EDIT){
            toggleCondRendAnimState(addMode, setAddMode);
        }
        if(categoryFn === CategoryFnMode.DELETE){
            setCategoryFn(CategoryFnMode.LIST);
        }else{
            setCategoryFn(CategoryFnMode.DELETE);
        }
    }

    return <DayHeaderRow className='noselect'>
    <DayH1>{curDay.year}년 {curDay.month + 1}월 {curDay.day}{curDay.day && '일'}</DayH1>
    <DayFn>
        {headerMode === TodoListMode.CategoryGeneral && <>
            <DayFnBtn onClick={() => toggleCategoryPanel()}>
                {<span>돌아가기</span>}
            </DayFnBtn>
            <DayFnBtn onClick={categoryAddToggleHandler}>
                {categoryFn === CategoryFnMode.ADD ? <>
                    <span>추가</span><span>완료</span>
                </> : <>
                    <span>카테고리</span><span>추가</span>
                </>}
            </DayFnBtn>
            <DayFnBtn onClick={categoryEditToggleHandler}>
                {categoryFn === CategoryFnMode.EDIT ? <>
                    <span>수정</span><span>완료</span>
                </> : <>
                    <span>카테고리</span><span>수정</span>
                </>}
            </DayFnBtn>
            <DayFnBtn onClick={categoryDeleteToggleHandler}>
                {categoryFn === CategoryFnMode.DELETE ? <>
                    <span>삭제</span><span>완료</span>
                </> : <>
                    <span>카테고리</span><span>삭제</span>
                </>}
            </DayFnBtn>
        </>}
        {headerMode === TodoListMode.TodoGeneral && <>
            <DayFnBtn onClick={() => toggleCategoryPanel()}>   
                <span>카테고리</span><span>관리</span>
            </DayFnBtn>
            <DayFnBtn onClick={() => toggleCondRendAnimState(addMode, setAddMode)}>
                {addMode.isMounted && addMode.showElem ? <>
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
            <DayFnBtn onClick={() => setHeaderMode(TodoListMode.TodoFunctional)}>
                <span>추가</span><span>기능</span>
            </DayFnBtn>
            
        </>}
        {headerMode === TodoListMode.TodoFunctional && <>
            <DayFnBtn onClick={() => dispatch(postponeTodo({ date: GetDjangoDateByCalTodoDay(curDay), postponeDayNum: DayDiffCalTodoDay(curDay, TOMORROW)}))}>
                <span>미완료</span><span>내일로</span>
            </DayFnBtn>
            <DayFnBtn onClick={() => dispatch(postponeTodo({ date: GetDjangoDateByCalTodoDay(curDay), postponeDayNum: 1}))}>
                <span>미완료</span><span>다음날로</span>
            </DayFnBtn>
            <DayFnBtn onClick={() => setHeaderMode(TodoListMode.TodoGeneral)}>
                <span>기능</span><span>끄기</span>
            </DayFnBtn>
        </>}
    </DayFn>
</DayHeaderRow>
}

export const DailyTodo = ({ curDay, setCurDay }: DailyTodoProps) => {
  const dispatch = useDispatch<AppDispatch>();
  const { elements, categories } = useSelector(selectTodo);

  // Header Mode
  const [headerMode, setHeaderMode_] = useState<TodoListMode>(TodoListMode.TodoGeneral);

  // Todo List ----------------------------------------------------------------------------------
  const [addMode, setAddMode] = useState<CondRendAnimState>(defaultCondRendAnimState);
  const [categorySort, setCategorySort] = useState<boolean>(true);

  const resetTodoListState = () => {
    setAddMode(defaultCondRendAnimState);
    setEditID(-1);
    setCategorySort(true);
  };

  // Category List -------------------------------------------------------------------------------
  const [categoryFn, setCategoryFn] = useState<CategoryFnMode>(CategoryFnMode.LIST);
  // const [addMode, setAddMode] : also used in Category List!
  // Category List - TodoCategory Create
  
  const resetCategoryListState = () => {
    setAddMode(defaultCondRendAnimState);
    setCategoryFn(CategoryFnMode.LIST);
  };

  // Todo Functional -----------------------------------------------------------------------------
  const [editID, setEditID] = useState<number>(-1);

  const resetTodoFunctionalState = () => {
    setAddMode(defaultCondRendAnimState);
    setEditID(-1);
  };

  const setHeaderMode = (target: TodoListMode) => {
    switch(target){
        case TodoListMode.TodoGeneral:
            resetTodoListState();
            break;
        case TodoListMode.CategoryGeneral:
            resetCategoryListState();
            break;
        case TodoListMode.TodoFunctional:
            resetTodoFunctionalState();
            break;
    };
    setHeaderMode_(target);
  };

  const setEditMode = () => {
    if(addMode.isMounted && addMode.showElem)
        return;
    toggleCondRendAnimState(addMode, setAddMode);
  };

  const editCompleteHandler = () => {
    setEditID(-1);
    setAddMode(defaultCondRendAnimState);
  };
  
  useEffect(() => {
    setHeaderMode(TodoListMode.TodoGeneral);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [curDay.day]);

  return <DailyTodoWrapper>
    <DailyTodoHeader headerMode={headerMode} setHeaderMode={setHeaderMode} addMode={addMode} setAddMode={setAddMode}
                     curDay={curDay} setCurDay={setCurDay} categoryFn={categoryFn} setCategoryFn={setCategoryFn}
                     categorySort={categorySort} setCategorySort={setCategorySort} setEditID={setEditID}
    />
    <DayBodyRow>
        {headerMode === TodoListMode.CategoryGeneral && <div>
            {categoryFn === CategoryFnMode.ADD && <CategoryAdder addMode={addMode} setAddMode={setAddMode} />}
            {categoryFn === CategoryFnMode.EDIT && <CategoryEditor  addMode={addMode} setAddMode={setAddMode} editObj={categories.find((e) => e.id === editID)!} editCompleteHandler={editCompleteHandler}/>}
            <TodoElementList style={addMode.showElem && addMode.isMounted ? { transform: "translateY(55px)" } : { transform: "translateY(0px)" }}>
                {categories.map((categ) => <TodoCategoryItem key={categ.id}>
                        <TodoElementColorCircle color={categ.color} ishard='false' />
                        <span>{categ.name}</span>
                        <div>
                            {categ.tag.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
                        </div>
                        {categoryFn === CategoryFnMode.DELETE && <div className='clickable' onClick={() => {
                            if (window.confirm('정말 카테고리를 삭제하시겠습니까?')) {
                                dispatch(deleteTodoCategory(categ.id));
                            }}}>
                            삭제
                        </div>}
                </TodoCategoryItem>)}
            </TodoElementList>
        </div>}
        {headerMode !== TodoListMode.CategoryGeneral && <>
            {addMode.showElem && ( headerMode === TodoListMode.TodoGeneral ? 
                (<TodoAdder addMode={addMode} setAddMode={setAddMode} curDay={curDay} />)
            :
                (elements[curDay.day!].find(((e) => e.id === editID)) && <TodoEditor addMode={addMode} setAddMode={setAddMode} curDay={curDay} editObj={elements[curDay.day!].find(((e) => e.id === editID))!} editCompleteHandler={editCompleteHandler}/>)
            )}
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
                                    .filter((todo) => todo.id !== editID)
                                    .map((todo) => {
                                        return <TodoItem key={todo.id} todo={todo} setEditMode={setEditMode}
                                                        fnMode={headerMode === TodoListMode.TodoFunctional} editID={editID} setEditID={setEditID}/>
                                })}
                                </TodoCategoryBody>
                            </TodoCategoryWrapper>
                        })}
                </>}
                {!categorySort && <>
                    {curDay.day && elements[curDay.day] && [...elements[curDay.day]] // For Read Only Array Sort, We have to copy that.
                    .sort((a, b) => b.priority - a.priority) // Descending Order! High Priority means Important Job.
                    .filter((todo) => todo.id !== editID)
                    .map((todo) => {
                        return <TodoItem key={todo.id} todo={todo} setEditMode={setEditMode}
                                        fnMode={headerMode === TodoListMode.TodoFunctional}  editID={editID} setEditID={setEditID}/>
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