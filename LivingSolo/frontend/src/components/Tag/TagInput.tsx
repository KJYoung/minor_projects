import React, { useEffect, useState } from 'react';
import { styled } from 'styled-components';

import { RoundButton } from '../../utils/Button';

import { TagBubbleCompact } from '../general/TagBubble';
import { useSelector } from 'react-redux';
import { TagElement, selectTag } from '../../store/slices/tag';
import { TagDialog } from './TagDialog';
import { DEFAULT_OPTION } from '../../utils/Constants';

// const NEW_OPTION = '$NEW$';
// const SEARCH_OPTION = '$SEARCH$';

interface TagInputProps {
  tags: TagElement[],
  setTags: React.Dispatch<React.SetStateAction<TagElement[]>>
}

// Tag Input Container.
export const TagInputForTrxnInput = ({ tags, setTags }: TagInputProps) => {
  const { elements }  = useSelector(selectTag);
  const [tagClassSelect, setTagClassSelect] = useState<string>(DEFAULT_OPTION); // Tag Class select value
  const [open, setOpen] = React.useState<boolean>(false);
  
  useEffect(() => {
    setTags([]);
  }, [elements, setTags]);

  const handleClickOpen = () => {
    setOpen(true);
  };
  const handleClose = () => {
    setOpen(false);
  };

  return (
    <TagInputDiv>
      <div>
        {tags.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
      </div>
      <RoundButton onClick={handleClickOpen}>+</RoundButton>
      <TagDialog open={open} handleClose={handleClose}
                  tags={tags} setTags={setTags} tagClassSelect={tagClassSelect} setTagClassSelect={setTagClassSelect} tag_max_length={5}/>
    </TagInputDiv>
  );
};

const TagInputDiv = styled.div`
    background-color: var(--ls-gray_lighter2);
    border: 1px solid var(--ls-gray_lighter);
    padding: 5px;
    border-radius: 5px;
    
    display: grid;
    place-items: center;
    grid-template-columns: 6fr 1fr;

    > div:first-child {
      > button {
        margin-right: 5px;
      }
    }
`;

// Tag Input Container for GridHeader
interface GridTagInputProps extends TagInputProps {
  closeHandler: () => void
};

export const TagInputForGridHeader = ({ tags, setTags, closeHandler }: GridTagInputProps) => {
  const { elements }  = useSelector(selectTag);
  const [tagClassSelect, setTagClassSelect] = useState<string>(DEFAULT_OPTION); // Tag Class select value
  const [open, setOpen] = React.useState<boolean>(false);
  
  useEffect(() => {
    setTags([]);
  }, [elements, setTags]);

  const handleClickOpen = () => {
    setOpen(true);
  };
  const handleClose = () => {
    setOpen(false);
  };

  return (
    <GridTagInputDiv>
      <div>
        {tags.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
      </div>
      <GridTagOpenBtn onClick={handleClickOpen}>+</GridTagOpenBtn>
      <GridTagCap>
        <span onClick={closeHandler}>Tag Filter...</span>
        <span onClick={closeHandler}>x</span>
      </GridTagCap>
      <TagDialog open={open} handleClose={handleClose}
                  tags={tags} setTags={setTags} tagClassSelect={tagClassSelect} setTagClassSelect={setTagClassSelect} tag_max_length={3}/>
    </GridTagInputDiv>
  );
};

const GridTagInputDiv = styled.div`
    border: 1px solid var(--ls-gray_lighter);
    padding: 10px 5px 0px 5px;
    border-radius: 5px;
    
    display: grid;
    grid-template-columns: 9fr 1fr;

    width: 100%;
    height: 100%;

    position: relative;

    > div:first-child {
      > button {
        margin-right: 5px;
      }
    }
`;

const GridTagOpenBtn = styled.span`
  cursor: pointer;
  color: var(--ls-blue);
  font-size: 22px;
  font-weight: 700;
`;
const GridTagCap = styled.div`
  position: absolute;
  top: -10px;
  font-size: 13px;
  
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  > span {
    background-color: white;
    margin-left: 10px;
    padding: 0px 3px 0px 3px;
    cursor: pointer;
    color: var(--ls-blue);
  };
  > span:first-child {
    margin-left: 5px;
    font-size: 13px;
  };
  > span:nth-child(2) {
    
  };
  > span:last-child {
    margin-right: 15px;
    color: var(--ls-red);
    font-size: 20px;
  };
`;

export const TagInputForTodo = ({ tags, setTags, closeHandler }: GridTagInputProps) => {
  const [tagClassSelect, setTagClassSelect] = useState<string>(DEFAULT_OPTION); // Tag Class select value
  const [open, setOpen] = React.useState<boolean>(false);

  const handleClickOpen = () => {
    setOpen(true);
  };
  const handleClose = () => {
    setOpen(false);
  };

  return (
    <TodoInputDiv>
      <div>
        {tags.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
      </div>
      <TodoTagOpenBtn onClick={handleClickOpen}>+</TodoTagOpenBtn>
      <TodoTagCap>
        <span onClick={closeHandler}>Tag</span>
      </TodoTagCap>
      <TagDialog open={open} handleClose={handleClose}
                  tags={tags} setTags={setTags} tagClassSelect={tagClassSelect} setTagClassSelect={setTagClassSelect} tag_max_length={5}/>
    </TodoInputDiv>
  );
};

export const TagInputForTodoCategory = ({ tags, setTags, closeHandler }: GridTagInputProps) => {
  const [tagClassSelect, setTagClassSelect] = useState<string>(DEFAULT_OPTION); // Tag Class select value
  const [open, setOpen] = React.useState<boolean>(false);

  const handleClickOpen = () => {
    setOpen(true);
  };
  const handleClose = () => {
    setOpen(false);
  };

  return (
    <TodoInputDiv>
      <div>
        {tags.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
      </div>
      <TodoTagOpenBtn onClick={handleClickOpen}>+</TodoTagOpenBtn>
      <TodoTagCap>
        <span onClick={closeHandler}>Tag</span>
      </TodoTagCap>
      <TagDialog open={open} handleClose={handleClose}
                  tags={tags} setTags={setTags} tagClassSelect={tagClassSelect} setTagClassSelect={setTagClassSelect} tag_max_length={3}/>
    </TodoInputDiv>
  );
};

const TodoInputDiv = styled.div`
  border: 1px solid var(--ls-gray_lighter);
  padding: 12px 0px 5px 5px;
  border-radius: 8px;
  margin-right: 10px;
  
  display: grid;
  grid-template-columns: 12fr 1fr;

  position: relative;

  > div:first-child {
    > button {
      margin-right: 5px;
    }
  }
`;
const TodoTagOpenBtn = styled.span`
  cursor: pointer;
  color: var(--ls-blue);
  font-size: 24px;
  font-weight: 500;
`;
const TodoTagCap = styled.div`
  position: absolute;
  top: -5px;
  left: 10px;
  font-size: 14px;
  
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  > span {
    background-color: white;
    margin-left: 10px;
    padding: 0px 3px 0px 3px;
    cursor: pointer;
    color: var(--ls-blue);
  };
  > span:first-child {
    margin-left: 5px;
    font-size: 13px;
  };
`;