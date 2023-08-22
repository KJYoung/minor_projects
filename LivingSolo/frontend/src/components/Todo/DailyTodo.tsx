import React, { useEffect, useState } from 'react';
import { styled } from 'styled-components';
import { CalTodoDay } from '../../utils/DateTime';
import { useSelector } from 'react-redux';
import { TodoElement, selectTodo } from '../../store/slices/todo';
import { TodoFastAdder, TodoItem } from './TodoItem';
import { CondRendAnimState, toggleCondRendAnimState, defaultCondRendAnimState } from '../../utils/Rendering';
import { TodoAdder, TodoEditor } from './TodoAdder';
import { CategoryAdder, CategoryEditor } from './CategoryAdder';
import { DailyTodoHeader } from './DailyTodoHeader';
import { TodoCategoryItem } from './CategoryItem';

export enum TodoFnMode {
    TodoGeneral, CategoryGeneral, TodoFunctional,
};
export enum CategoryFnMode {
    LIST, ADD, EDIT, DELETE
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

export const DailyTodo = ({ curDay, setCurDay }: DailyTodoProps) => {
  const { elements, categories } = useSelector(selectTodo);

  // Header Mode
  const [headerMode, setHeaderMode_] = useState<TodoFnMode>(TodoFnMode.TodoGeneral);

  // Todo List ----------------------------------------------------------------------------------
  const [addMode, setAddMode] = useState<CondRendAnimState>(defaultCondRendAnimState);
  const [fastAddCategID, setFastAddCategID] = useState<number>(-1);
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

  const setHeaderMode = (target: TodoFnMode) => {
    switch(target){
        case TodoFnMode.TodoGeneral:
            resetTodoListState();
            break;
        case TodoFnMode.CategoryGeneral:
            resetCategoryListState();
            break;
        case TodoFnMode.TodoFunctional:
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
    setCategoryFn(CategoryFnMode.LIST);
  };
  
  useEffect(() => {
    setHeaderMode(TodoFnMode.TodoGeneral);
    // eslint-disable-next-line react-hooks/exhaustive-deps
}, [curDay.day]);

  useEffect(() => {
    setFastAddCategID(-1);
  }, [curDay.day, headerMode]);

  return <DailyTodoWrapper>
    <DailyTodoHeader headerMode={headerMode} setHeaderMode={setHeaderMode} addMode={addMode} setAddMode={setAddMode}
                     curDay={curDay} setCurDay={setCurDay} categoryFn={categoryFn} setCategoryFn={setCategoryFn}
                     categorySort={categorySort} setCategorySort={setCategorySort} setEditID={setEditID}
    />
    <DayBodyRow>
        {headerMode === TodoFnMode.CategoryGeneral && <div>
            {categoryFn === CategoryFnMode.ADD && <CategoryAdder addMode={addMode} setAddMode={setAddMode} />}
            {categoryFn === CategoryFnMode.EDIT && categories.find((e) => e.id === editID) && <CategoryEditor  addMode={addMode} setAddMode={setAddMode} editObj={categories.find((e) => e.id === editID)!} editCompleteHandler={editCompleteHandler}/>}
            <TodoElementList style={addMode.showElem && addMode.isMounted ? { transform: "translateY(55px)" } : { transform: "translateY(0px)" }}>
                {categories.map((categ) => <TodoCategoryItem category={categ} categoryFn={categoryFn} editID={editID} setEditID={setEditID} key={categ.id}/>)}
            </TodoElementList>
        </div>}
        {headerMode !== TodoFnMode.CategoryGeneral && <>
            {addMode.showElem && ( headerMode === TodoFnMode.TodoGeneral ? 
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
                                    {
                                        fastAddCategID === categoryElement.id ?
                                            <span className='clickable noselect' onClick={() => setFastAddCategID(-1)}>-</span>
                                        :  
                                            <span className='clickable noselect' onClick={() => setFastAddCategID(categoryElement.id)}>+</span>
                                    }
                                </TodoCategoryHeader>
                                <TodoCategoryBody>
                                    {fastAddCategID === categoryElement.id && 
                                        <TodoFastAdder categ={categories.find((e) => e.id === categoryElement.id)!} addCompleteHandler={() => setFastAddCategID(-1)} curDay={curDay} />
                                    }
                                    {categoryElement.todos // For Read Only Array Sort, We have to copy that.
                                        .sort((a, b) => b.priority - a.priority) // Descending Order! High Priority means Important Job.
                                        .filter((todo) => todo.id !== editID)
                                        .map((todo) => {
                                            return <TodoItem key={todo.id} todo={todo} setEditMode={setEditMode}
                                                            fnMode={headerMode === TodoFnMode.TodoFunctional} editID={editID} setEditID={setEditID}/>
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
                                        fnMode={headerMode === TodoFnMode.TodoFunctional}  editID={editID} setEditID={setEditID}/>
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

const DayBodyRow = styled.div`
    width: 100%;
    margin-top: 20px;

    position: relative;
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
    span:last-child{
        margin-left: 10px;
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