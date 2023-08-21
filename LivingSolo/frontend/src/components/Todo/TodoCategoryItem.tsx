import React from 'react';
import { styled } from 'styled-components';
import { useDispatch } from 'react-redux';
import { TodoCategory, deleteTodoCategory } from '../../store/slices/todo';
import { AppDispatch } from '../../store';
import { TagBubbleCompact } from '../general/TagBubble';
import { CategoryFnMode } from './DailyTodo';

interface TodoCategoryItemProps {
  category: TodoCategory,
  categoryFn: CategoryFnMode,
  editID: number,
  setEditID: React.Dispatch<React.SetStateAction<number>>,
};

export const TodoCategoryItem = ({ category, categoryFn, editID, setEditID }: TodoCategoryItemProps) => {
  const dispatch = useDispatch<AppDispatch>();

  const editHandler = (id: number) => {
    setEditID(id);
  };

  const deleteHandler = (id: number) => {
    if (window.confirm('정말 카테고리를 삭제하시겠습니까?')) {
        dispatch(deleteTodoCategory(id));
    }
  };

  return <TodoCategoryItemWrapper key={category.id}>
    <TodoElementColorCircle color={category.color} ishard='false' />
        <span>{category.name}</span>
        <div>
            {category.tag.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
        </div>
        {
            categoryFn === CategoryFnMode.EDIT && 
            <div className={editID !== category.id ? 'clickable' : ''} onClick={() => editHandler(category.id)}>
                수정
            </div>
        }
        {
            categoryFn === CategoryFnMode.DELETE && 
                <div className='clickable' onClick={() => deleteHandler(category.id)}>
                    삭제
                </div>
        }
    </TodoCategoryItemWrapper>
};

const TodoCategoryItemWrapper = styled.div`
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

// const TodoElementWrapper = styled.div`
//     width: 100%;
//     padding: 10px;
//     border-bottom: 0.5px solid green;
    
//     display: grid;
//     grid-template-columns: 1fr 10fr 5fr 4fr;
//     align-items: center;

//     &:last-child {
//         border-bottom: none;
//     };
// `;

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