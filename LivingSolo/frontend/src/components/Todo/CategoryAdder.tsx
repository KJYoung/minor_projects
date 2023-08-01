import { styled } from "styled-components";
import { CondRendAnimState, condRendMounted, condRendUnmounted, onAnimEnd } from "../../utils/Rendering";
import { useState } from "react";
import { TagElement } from "../../store/slices/tag";
import { TodoCategoryCreateReqType, createTodoCategory } from "../../store/slices/todo";
import { useDispatch } from "react-redux";
import { TagInputForTodoCategory } from "../Trxn/TagInput";
import { AppDispatch } from "../../store";
import { CalTodoDay } from "../../utils/DateTime";
import { getRandomHex } from "../../styles/color";


interface TodoAdderProps {
    addMode: CondRendAnimState,
    setAddMode: React.Dispatch<React.SetStateAction<CondRendAnimState>>,
    curDay: CalTodoDay,
};

const todoCategorySkeleton = {
    name: '',
    color: '#000000',
}

export const CategoryAdder = ({ addMode, setAddMode, curDay } : TodoAdderProps) => {
    const dispatch = useDispatch<AppDispatch>();

    // Todo List - Create
    const [categTags, setCategTags] = useState<TagElement[]>([]);
    const [newTodoCategory, setNewTodoCategory] = useState<TodoCategoryCreateReqType>({...todoCategorySkeleton, tag: categTags});

    return <CategoryAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted} onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
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
    
</CategoryAdderWrapper>
};

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

const CategoryAdderWrapper = styled.div`
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