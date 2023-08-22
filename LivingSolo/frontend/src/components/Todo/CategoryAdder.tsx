import { styled } from "styled-components";
import { CondRendAnimState, condRendMounted, condRendUnmounted, onAnimEnd } from "../../utils/Rendering";
import { useEffect, useState } from "react";
import { TagElement } from "../../store/slices/tag";
import { TodoCategory, TodoCategoryCreateReqType, TodoCategoryEditReqType, createTodoCategory, editTodoCategory } from "../../store/slices/todo";
import { useDispatch } from "react-redux";
import { TagInputForTodoCategory } from "../Tag/TagInput";
import { AppDispatch } from "../../store";
import { getRandomHex } from "../../styles/color";
import { ColorDialog } from "../general/ColorDialog";


interface CategoryAdderProps {
    addMode: CondRendAnimState,
    setAddMode: React.Dispatch<React.SetStateAction<CondRendAnimState>>,
};

interface CategoryEditorProps extends CategoryAdderProps {
    editObj: TodoCategory,
    editCompleteHandler: () => void,
};

const todoCategorySkeleton = {
    name: '',
    color: '#000000',
}

export const CategoryAdder = ({ addMode, setAddMode } : CategoryAdderProps) => {
    const dispatch = useDispatch<AppDispatch>();

    // Category List - Create
    const [categTags, setCategTags] = useState<TagElement[]>([]);
    const [newTodoCategory, setNewTodoCategory] = useState<TodoCategoryCreateReqType>({...todoCategorySkeleton, tag: categTags});

    // Color Dialog
    const [open, setOpen] = useState<boolean>(false);
    const [color, setColor] = useState<string>(newTodoCategory.color);

    const handleClose = () => {
        setOpen(false);
    };
    const colorDialogOpenHandler = () => {
        setOpen(true);
    };

    return <CategoryAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted} onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
    <CategoryAdderRow>
        <CategoryColorCircle color={color} ishard={'false'}>
                <div onClick={() => { setColor(getRandomHex()); }}></div>
                <div className="clickable" onClick={colorDialogOpenHandler}>...</div>
        </CategoryColorCircle>
        <TagInputForTodoCategory tags={categTags} setTags={setCategTags} closeHandler={() => {}}/>
        <CategoryAdderInputWrapper>
            <input type="text" placeholder='Category Name' value={newTodoCategory.name} onChange={(e) => setNewTodoCategory((nTC) => { return {...nTC, name: e.target.value}})}/>
            <button onClick={() => { 
                dispatch(createTodoCategory({...newTodoCategory, tag: categTags, color }));
                setCategTags([]);
                setNewTodoCategory({...todoCategorySkeleton, tag: categTags});
            }}>Create</button>
        </CategoryAdderInputWrapper>
    </CategoryAdderRow>
    <ColorDialog open={open} handleClose={handleClose}
                 color={color} setColor={setColor}/>
</CategoryAdderWrapper>
};

export const CategoryEditor = ({ addMode, setAddMode, editObj, editCompleteHandler } : CategoryEditorProps) => {
    const dispatch = useDispatch<AppDispatch>();

    useEffect(() => {
        setCategTags(editObj.tag);
        setEditCategory({...editObj});
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [editObj.id]);

    // Category List - Edit
    const [categTags, setCategTags] = useState<TagElement[]>(editObj.tag);
    const [editCategory, setEditCategory] = useState<TodoCategoryEditReqType>({...editObj});

    // Color Dialog
    const [open, setOpen] = useState<boolean>(false);
    const [color, setColor] = useState<string>(editCategory.color);

    const handleClose = () => {
        setOpen(false);
    };
    const colorDialogOpenHandler = () => {
        setOpen(true);
    };

    return <CategoryAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted} onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
        <CategoryAdderRow>
            <CategoryColorCircle color={color} ishard={'false'}>
                    <div onClick={() => { setColor(getRandomHex()); }}></div>
                    <div className="clickable" onClick={colorDialogOpenHandler}>...</div>
            </CategoryColorCircle>
            <TagInputForTodoCategory tags={categTags} setTags={setCategTags} closeHandler={() => {}}/>
            <CategoryAdderInputWrapper>
                <input type="text" placeholder='Category Name' value={editCategory.name} onChange={(e) => setEditCategory((eC) => { return {...eC, name: e.target.value}})}/>
                <button onClick={() => { 
                    dispatch(editTodoCategory({...editCategory, tag: categTags, color}));
                    editCompleteHandler();
                }}>Edit</button>
            </CategoryAdderInputWrapper>
        </CategoryAdderRow>
        <ColorDialog open={open} handleClose={handleClose}
                 color={color} setColor={setColor}/>
    </CategoryAdderWrapper>
};

const CategoryColorCircle = styled.div<{ color: string, ishard: string }>`
    position: relative;
    width: 20px;
    height: 20px;

    > div {
        cursor: pointer;
    }

    > div:first-child {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: ${props => ((props.ishard === 'true') ? '2px solid var(--ls-red)' : 'none')};
        background-color: ${props => (props.color)};;
        
        margin-right: 10px;

        display: flex;
        justify-content: center;
        align-items: center;
    }

    > div:last-child {
        position: absolute;
        top: 10px;
        left: 20px;
        width: 20px;
        height: 20px;
    }
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