/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { TagElement, fetchTagDetail, fetchTagPresets, fetchTags, fetchTagsIndex, selectTag } from '../store/slices/tag';
import { IPropsActive } from '../utils/Interfaces';
import { CondRendAnimState, defaultCondRendAnimState, toggleCondRendAnimState } from '../utils/Rendering';
import { TagDetail } from '../components/Tag/TagDetail';
import { TagList } from '../components/Tag/TagList';

export enum TagViewerMode {
  TagClass, Tag, TagPreset
};

const TagMain = () => {
  const [addMode, setAddMode] = useState<CondRendAnimState>(defaultCondRendAnimState);
  const [tagViewerMode, setTagViewerMode] = useState<TagViewerMode>(TagViewerMode.TagClass);
  const [selectedTag, setSelectedTag] = useState<TagElement | undefined>();

  const dispatch = useDispatch<AppDispatch>();

  const { errorState } = useSelector(selectTag);

  // Fetch Tag Related Things!
  useEffect(() => {
    dispatch(fetchTags());
    dispatch(fetchTagsIndex());
    dispatch(fetchTagPresets());
    setSelectedTag(undefined);
  }, [dispatch, errorState]);

  useEffect(() => {
    if(selectedTag){
      dispatch(fetchTagDetail(selectedTag.id));
    }
  }, [dispatch, selectedTag]);

  const classAddToggleHandler = () => {
    toggleCondRendAnimState(addMode, setAddMode); // ON
  };

  const tabChangeHandler = (newTabState : TagViewerMode) => {
    if(newTabState === tagViewerMode)
      return;
    if(addMode.showElem)
      classAddToggleHandler();
    setTagViewerMode(newTabState);
  };

  const tagSelectHandler = (tag: TagElement) => {
    setSelectedTag(tag);
  };

  return (
    <Wrapper>
      <InnerWrapper>
        <LeftWrapper>
          <ListWrapper>
            <TagTabHeader>
              <div>
                <TagTabName active={(tagViewerMode === TagViewerMode.TagClass).toString()} onClick={() => tabChangeHandler(TagViewerMode.TagClass)}>TagClass</TagTabName>
                <TagTabName active={(tagViewerMode === TagViewerMode.Tag).toString()}  onClick={() => tabChangeHandler(TagViewerMode.Tag)}>Tag</TagTabName>
                <TagTabName active={(tagViewerMode === TagViewerMode.TagPreset).toString()} onClick={() => tabChangeHandler(TagViewerMode.TagPreset)}>TagPreset</TagTabName>
              </div>
              <div>
                <span className='clickable' onClick={classAddToggleHandler}>+</span>
              </div>
            </TagTabHeader>
            
            <TagList addMode={addMode} setAddMode={setAddMode} tagViewerMode={tagViewerMode} setTagViewerMode={setTagViewerMode} tagSelectHandler={tagSelectHandler}/>
          </ListWrapper>
        </LeftWrapper>
        <RightWrapper>
          <TagDetail selectedTag={selectedTag}/>
        </RightWrapper>
      </InnerWrapper>
    </Wrapper>
  );
};

export default TagMain;

const Wrapper = styled.div`
  width: 100%;
  height: 100%;
  min-height: 100vh;
  background-color: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: start;
  align-items: start;
`;

const InnerWrapper = styled.div`
  width: 100%;
  height: 92%;
  display: flex;
  flex-direction: row;
  justify-content: start;
  align-items: start;
`;

const LeftWrapper = styled.div`
  width: 1500px;
  height: 100%;
  padding: 30px 0px 0px 30px;

  display: flex;
  flex-direction: column;
`;

const RightWrapper = styled.div`
  width: 100%;
  height: 100%;
  padding: 30px 30px 0px 15px;

  display: flex;
  flex-direction: column;

  > div {
    width: 100%;
    height: 100%;
  }
`;

const ListWrapper = styled.div`
  width: 100%;
`;

const TagTabHeader = styled.div`
  font-size: 28px;
  padding: 4px;
  border-bottom: 1px solid black;

  display: grid;
  grid-template-columns: 8fr 2fr;
  
  > div:first-child {
    display: flex;
    justify-content: space-between;
  }

  > div:last-child {
    display: flex;
    justify-content: center;
  }
`;

const TagTabName = styled.span<IPropsActive>`
  color: ${props => ((props.active === 'true') ? 'var(--ls-blue)' : 'var(--ls-gray)')};
  cursor: ${props => ((props.active === 'true') ? 'default' : 'pointer')};
  &:hover { 
    ${({ active }) => active === 'false' && `color: var(--ls-green)`}
  }
`;

