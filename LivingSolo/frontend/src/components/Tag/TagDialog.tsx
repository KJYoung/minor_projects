import { useSelector } from "react-redux";
import { TagElement, selectTag } from "../../store/slices/tag";
import { useState } from "react";
import { BootstrapDialog, BootstrapDialogTitle } from "../general/Dialog";
import { styled } from "styled-components";
import { Button, DialogActions, DialogContent } from "@mui/material";
import { TagBubbleCompactPointer, TagBubbleWithFunc, TagBubbleX } from "../general/TagBubble";

interface TagDialogProps {
    open: boolean,
    handleClose: () => void,
    tags: TagElement[],
    setTags: React.Dispatch<React.SetStateAction<TagElement[]>>,
    tagClassSelect: string,
    setTagClassSelect: React.Dispatch<React.SetStateAction<string>>,
    tag_max_length: number,
};

const DEFAULT_OPTION = '$NONE$';


export const TagDialog = ({open, handleClose, tags, setTags, tagClassSelect, setTagClassSelect, tag_max_length} : TagDialogProps) => {
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
};


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

const TagInputClearSpan = styled.span<{ active: string }>`
    margin-top: 3px;
    font-size: 17px;
    color: ${props => ((props.active === 'true') ? 'var(--ls-blue)' : 'var(--ls-gray)')};
    background-color: transparent;
    cursor: ${props => ((props.active === 'true') ? 'pointer' : 'default')};;
    margin-left: 20px;
`;
