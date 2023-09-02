/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect } from 'react';
import styled from 'styled-components';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { fetchTags, fetchTagsIndex, selectTag } from '../store/slices/tag';
import { TagBubbleCompact } from '../components/general/TagBubble';

const TagMain = () => {
  const dispatch = useDispatch<AppDispatch>();

  const { elements, index, errorState } = useSelector(selectTag);

  // Fetch Tag Related Things!
  useEffect(() => {
      dispatch(fetchTags());
      dispatch(fetchTagsIndex());
    }, [dispatch, errorState]);

  return (
    <Wrapper>
      <InnerWrapper>
        <LeftWrapper>
            <span>TagCategory</span>
            {elements.map((tagClass) => {
                return <div key={tagClass.id}>
                    <span>Tag Class Name : {tagClass.name}</span>
                    <span>---</span>
                    <div>
                        {tagClass.tags?.map((tag) => {
                            return <TagBubbleCompact color={tag.color}>{tag.name}</TagBubbleCompact>
                        })}
                    </div>
                </div>
            })}
            <span>Tag</span>
            {index.map((tag) => {
                return <TagBubbleCompact color={tag.color}>{tag.name}</TagBubbleCompact>
            })}
            <span>TagPreset</span>
            <span>Not Yet</span>
        </LeftWrapper>
        <RightWrapper></RightWrapper>
        
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
  width: 900px;
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