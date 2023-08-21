import { useDispatch, useSelector } from "react-redux";
import { CalTodoDay, DayDiffCalTodoDay, GetDjangoDateByCalTodoDay, TODAY, TOMORROW } from "../../utils/DateTime";
import { CondRendAnimState, toggleCondRendAnimState } from "../../utils/Rendering";
import { CategoryFnMode, TodoFnMode } from "./DailyTodo";
import { AppDispatch } from "../../store";
import { postponeTodo, selectTodo } from "../../store/slices/todo";
import { notificationDefault } from "../../utils/sendNoti";
import { styled } from "styled-components";

interface DailyTodoHeaderProps {
    headerMode: TodoFnMode,
    setHeaderMode: (tLM : TodoFnMode) => void,
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

export const DailyTodoHeader = ({ 
                            headerMode, setHeaderMode, addMode, setAddMode, curDay, setCurDay,
                            categoryFn, setCategoryFn, categorySort, setCategorySort, setEditID,
                        }: DailyTodoHeaderProps) => 
{
    const dispatch = useDispatch<AppDispatch>();
    const { categories } = useSelector(selectTodo);

    const toggleCategoryPanel = () => {
        setHeaderMode(headerMode === TodoFnMode.CategoryGeneral ? TodoFnMode.TodoGeneral : TodoFnMode.CategoryGeneral);
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
        {headerMode === TodoFnMode.CategoryGeneral && <>
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
        {headerMode === TodoFnMode.TodoGeneral && <>
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
            <DayFnBtn onClick={() => setHeaderMode(TodoFnMode.TodoFunctional)}>
                <span>추가</span><span>기능</span>
            </DayFnBtn>
            
        </>}
        {headerMode === TodoFnMode.TodoFunctional && <>
            <DayFnBtn onClick={() => dispatch(postponeTodo({ date: GetDjangoDateByCalTodoDay(curDay), postponeDayNum: DayDiffCalTodoDay(curDay, TOMORROW)}))}>
                <span>미완료</span><span>내일로</span>
            </DayFnBtn>
            <DayFnBtn onClick={() => dispatch(postponeTodo({ date: GetDjangoDateByCalTodoDay(curDay), postponeDayNum: 1}))}>
                <span>미완료</span><span>다음날로</span>
            </DayFnBtn>
            <DayFnBtn onClick={() => setHeaderMode(TodoFnMode.TodoGeneral)}>
                <span>기능</span><span>끄기</span>
            </DayFnBtn>
        </>}
    </DayFn>
</DayHeaderRow>
};

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