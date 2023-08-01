import React, { useEffect, useState } from 'react';
import { styled } from 'styled-components';
import { styled as styledMUI } from '@mui/material/styles';
import { RoundButton } from '../../utils/Button';

import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import { TagBubbleCompact, TagBubbleCompactPointer, TagBubbleWithFunc, TagBubbleX } from '../general/TagBubble';
import { useSelector } from 'react-redux';
import { TagElement, selectTag } from '../../store/slices/tag';

interface TagDialogProps {
  open: boolean,
  handleClose: () => void,
  tags: TagElement[],
  setTags: React.Dispatch<React.SetStateAction<TagElement[]>>,
  tagClassSelect: string,
  setTagClassSelect: React.Dispatch<React.SetStateAction<string>>,
  tag_max_length: number,
}

const DEFAULT_OPTION = '$NONE$';
// const NEW_OPTION = '$NEW$';
// const SEARCH_OPTION = '$SEARCH$';

const TagDialog = ({open, handleClose, tags, setTags, tagClassSelect, setTagClassSelect, tag_max_length} : TagDialogProps) => {
  const { elements, index } = useSelector(selectTag);
  
  const [unfoldView, setUnfoldView] = useState<boolean>(true); // For convenience.
  const [tagSelect, setTagSelect] = useState<string>(DEFAULT_OPTION); // Tag select value

  const clearTagInput = () => {
    setTags([]);
    setTagClassSelect(DEFAULT_OPTION);
  };

  return <div>
    <BootstrapDialog
      onClose={handleClose}
      aria-labelledby="customized-dialog-title"
      open={open}
    >
      <DialogBody>
        <BootstrapDialogTitle id="customized-dialog-title" onClose={handleClose}>
          태그 설정
        </BootstrapDialogTitle>
        <DialogContent dividers>
          <SetTagHeaderWrapper>
            <SetTagHeader>
              설정된 태그{' '} 
              <TagLengthIndicator active={(tags.length >= tag_max_length).toString()}>{tags.length}</TagLengthIndicator> / {tag_max_length}
            </SetTagHeader>
            <TagInputClearSpan onClick={() => clearTagInput()}active={(tags.length !== 0).toString()}>Clear</TagInputClearSpan>
          </SetTagHeaderWrapper>
          <SetTagList>{tags?.map((ee) =>
            <TagBubbleWithFunc key={ee.id} color={ee.color}>
              {ee.name}
              <TagBubbleX onClick={() => setTags((tags) => tags.filter((t) => t.id !== ee.id))}/>
            </TagBubbleWithFunc>)}
          </SetTagList>
          <TagListWrapper>
            <TagListHeader>
              <span>태그 목록</span>
              <button onClick={() => setUnfoldView((ufV) => !ufV)}>{unfoldView ? '계층 구조로 보기' : '펼쳐 보기'}</button>
            </TagListHeader>
            <TagListBody>
              {unfoldView ? <>
              {/* Unfolded View */}
                {
                  index
                  .filter((tagElem) => { 
                    const tagsHasTagElem = tags.find((tag) => tag.id === tagElem.id);
                    return tagsHasTagElem === undefined; 
                  }) // Filtering Unselected!
                  .map((tagElem) =>
                    <TagBubbleCompactPointer onClick={() => setTags((tags) => (tags.length >= tag_max_length) ? tags : [...tags, tagElem])} key={tagElem.id} color={tagElem.color}>
                      {tagElem.name}
                    </TagBubbleCompactPointer>
                  )
                }
              </> 
              : <> 
              {/* Hierarchical View */}
                <select data-testid="tagSelect" value={tagClassSelect} onChange={(e) => setTagClassSelect(e.target.value)}>
                  <option disabled value={DEFAULT_OPTION}>
                    - 태그 클래스 -
                  </option>
                  {elements.map(tag => {
                      return (
                        <option value={tag.id} key={tag.id}>
                          {tag.name}
                        </option>
                      );
                    })}
                </select>
                <select data-testid="tagSelect2" value={tagSelect} onChange={(e) => {
                  const matchedElem = index.find((elem) => {
                    // console.log(`${elem.id.toString()} vs ${e.target.value}`);
                    return elem.id.toString() === e.target.value;
                  });
                  if(matchedElem)
                    setTags((array) => [...array, matchedElem]);
                  setTagSelect(DEFAULT_OPTION);
                }}>
                  <option disabled value={DEFAULT_OPTION}>
                    - 태그 이름 -
                  </option>
                  {elements.filter(tagClass => tagClass.id === Number(tagClassSelect))[0]?.tags?.map(tag => {
                    return (
                      <option value={tag.id} key={tag.id}>
                        {tag.name}
                      </option>
                    );
                  })}
                </select>
              </>
              }
            </TagListBody>
          </TagListWrapper>
        </DialogContent>
        <DialogActions>
          <Button autoFocus onClick={handleClose}>
            닫기
          </Button>
        </DialogActions>
      </DialogBody>
      
    </BootstrapDialog>
  </div>
}

const BootstrapDialog = styledMUI(Dialog)(({ theme }) => ({
  '& .MuiDialogContent-root': {
    padding: theme.spacing(2),
  },
  '& .MuiDialogActions-root': {
    padding: theme.spacing(1),
  },
}));

const DialogBody = styled.div`
  width: 500px;
`;
const SetTagHeaderWrapper = styled.div`
  display: flex;
  justify-content: space-between;
`;
const SetTagHeader = styled.span`
`;
const TagLengthIndicator = styled.span<{ active: string }>`
  color: ${props => ((props.active === 'true') ? 'var(--ls-red)' : 'var(--ls-blue)')};
`;

const SetTagList = styled.div`
  display: flex;
  flex-wrap: wrap;
  width: 100%;
  margin-top: 10px;
  min-height: 60px;
`;
const TagListWrapper = styled.div`
  display: flex;
  flex-direction: column;
  border-top: 1px solid var(--ls-gray_lighter);
  padding-top: 10px;
  margin-top: 10px;
`;
const TagListHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 15px;
`;
const TagListBody = styled.div`
  min-height: 50px;
`;
export interface DialogTitleProps {
  id: string;
  children?: React.ReactNode;
  onClose: () => void;
}

function BootstrapDialogTitle(props: DialogTitleProps) {
  const { children, onClose, ...other } = props;
  return (
    <DialogTitle sx={{ m: 0, p: 2 }} {...other}>
      {children}
      {onClose ? (
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{
            position: 'absolute',
            right: 8,
            top: 8,
            color: (theme) => theme.palette.grey[500],
          }}
        >
          <CloseIcon />
        </IconButton>
      ) : null}
    </DialogTitle>
  );
}

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

const TagInputClearSpan = styled.span<{ active: string }>`
    margin-top: 3px;
    font-size: 17px;
    color: ${props => ((props.active === 'true') ? 'var(--ls-blue)' : 'var(--ls-gray)')};
    background-color: transparent;
    cursor: ${props => ((props.active === 'true') ? 'pointer' : 'default')};;
    margin-left: 20px;
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