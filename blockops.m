% BLOCKOPS
%
% Examples:
%
%       a = [
%          1     1     3     3
%          1     1     3     3
%          2     2     4     4
%          2     2     4     4];
%
%
%      BLOCKOPS(a, [2 2])
%
%      ans =
%          1     2     3     4
%          1     2     3     4
%          1     2     3     4
%          1     2     3     4
%
%
%      BLOCKOPS(a, [2 2], 2)
%
%      ans =
%          1     1     1     1
%          2     2     2     2
%          3     3     3     3
%          4     4     4     4
%
%
%      BLOCKOPS(a, [2 2], 3)
%
%      ans(:,:,1) =
%          1     1
%          1     1
%
%      ans(:,:,2) =
%          2     2
%          2     2
%
%      ans(:,:,3) =
%          3     3
%          3     3
%
%      ans(:,:,4) =
%          4     4
%          4     4
%
%
%
%      BLOCKOPS(a, [2 2], 'split')
%
%      ans{1} =
%          1     1
%          1     1
%
%      ans{2} =
%          2     2
%          2     2
%
%      ans{3} =
%          3     3
%          3     3
%
%      ans{4} =
%          4     4
%          4     4
%
%
%
%
%      B = [2
%           3];
%
%      BLOCKOPS(a, [2 2], @mtimes, B)
%
%      ans =
%          5  10
%          5  10
%         15  20
%         15  20
%
%
%      BLOCKOPS(a, [2 2], @mtimes, B, 'split')
%
%      ans{1} =
%          5
%          5
%
%      ans{2} =
%          10
%          10
%
%      ans{3} =
%          15
%          15
%
%      ans{4} =
%          20
%          20
%
%
%      C = [1
%           4];   % expansion enabled (bsxfun)
%
%      BLOCKOPS(a, [2 2], @times, C, 'split')
%
%      ans{1} =
%           1      1
%           4      4
%
%      ans{2} =
%           2      2
%           8      8
%
%      ans{3} =
%           3      3
%          12     12
%
%      ans{4} =
%           4      4
%          16     16
%
%
%
% See also reshape, permute, kron.

function outArg = blockops(varargin)
    
    %% Initialize & handle simple cases
    
    % Basic checks
    argc = nargin;
    assert(argc >= 1, 'Not enough input arguments.');
    assert(nargout <= 1, 'Too many output arguments.');
    
    
    
    % Default values
    A = varargin{1};
    if isempty(A)
        error('blockops:invalid_array',...
            'BLOCKOPS can not operate on empty arrays.');
    end
    blockSize = [];
    dim       = 1;
    fcn       = [];
    B         = [];
    split     = false;
    
    
    if argc >= 2
        blockSize = varargin{2};        
        if ~(isnumeric(blockSize) && (...
                (isvector(blockSize) && all(isfinite(blockSize)) && all(blockSize>0) && all(round(blockSize)==blockSize)) || ...
                isempty(blockSize)))
            error('blockops:invalid_block_size',...
                'Input argument ''blockSize'' must be a vector of positive, finite, integer values.');
        end
        
        if numel(blockSize)==1
            blockSize = [blockSize blockSize]; end        
        blockSize = blockSize(1:find(blockSize~=1,1,'last'));
        if numel(blockSize) < ndims(A)
            blockSize = [blockSize ones(1,ndims(A)-numel(blockSize))]; end
        
        if ndims(A) < numel(blockSize) || any(blockSize > size(A)) || any(rem(size(A),blockSize))        
            if isvector(A) && numel(A) > 1
                nameA = 'vector';
            elseif isscalar(A)
                nameA = 'scalar';
            elseif ndims(A)==2
                nameA = 'matrix';
            else
                nameA = 'array';
            end
            error('blockops:invalid_block_size',...
                ['Block size not possible for given input ' nameA '.']);
        end
    end
    
    if argc >= 3
        arg = varargin{3};
        if isnumeric(arg) 
            if ~isempty(arg) 
                if isscalar(arg) && isfinite(arg) && arg > 0
                    dim = arg;
                else
                    error('blockops:invalid_input_argument',...
                        'Input argument ''DIM'' must be a finite, positive, scalar integer.');
                end
            else
                error('blockops:invalid_input_argument',...
                    'Argument ''DIM'' may not be empty.');
            end
            
        elseif ischar(arg)
            split = strcmpi(arg, 'split');
            if ~split
                error('blockops:invalid_input_argument',...
                    'Unrecognized string option: ''%s''.', arg);
            end
            
        elseif isa(arg, 'function_handle')
            fcn = arg;
            
        else
            error('blockops:invalid_input_argument',...
                'BLOCKOPS is not defined for a third input argument of type ''%s''.', class(arg));                        
        end
    end
    if argc <= 3 && (isempty(blockSize) || all(blockSize==1))
        % Call is equivalent to simple linearization:
        D = 1:max(dim, ndims(A));
        D(D==dim) = 1;  D(1) = dim;
        outArg = permute(A(:), D);
        if split
            outArg = num2cell(outArg); end
        return;
    end
      
    
    if argc >= 4
        arg = varargin{4};
        if ischar(arg)
            split = strcmpi(arg, 'split');
            if ~split
                error('blockops:invalid_input_argument',...
                    'Unrecognized string option: ''%s''.', arg);
            end
        else
            B = arg;            
        end
    end 
    if argc <= 4 && isempty(B) && ~isempty(fcn) && (isempty(blockSize) || all(blockSize==1))
        % Call is equivalent to ARRAYFUN:
        outArg = arrayfun(fcn, A, 'UniformOutput', false);
        if all(cellfun('prodofsize', outArg)==1)
            outArg = reshape([outArg{:}], size(outArg)); end      
        if split
            outArg = num2cell(outArg); end
        return;
    end
    
    
    if argc >= 5
        arg = varargin{5}; 
        if ischar(arg)
            split = strcmpi(arg, 'split');
            if ~split
                error('blockops:invalid_input_argument',...
                    'Unrecognized string option: ''%s''.', arg);
            end 
        else
            error('blockops:invalid_input_argument',...
                'BLOCKOPS is not defined for a fifth input argument of type ''%s''.', class(arg));                        
        end
    end
    if ~isempty(fcn) && ~isempty(B) && (isempty(blockSize) || all(blockSize==1))
        % Call is equivalent to BSXFUN: 
        try
            % FIXME: function with different output dimensions must be
            % captured in cell array
            outArg = bsxfun(fcn, A,B);
            if split
                outArg = num2cell(outArg); end
            return;
        catch ME
            ME2 = MException('blockops:function_error',...
                'Function failed to evaluate for given input arguments.');
            rethrow(addCause(ME2, ME))
        end        
    end
    
    %% Process blocks
    
    % Total number of blocks in A
    blocksPerDim = size(A)./blockSize;
    numBlocks    = prod(blocksPerDim); 
    
    firstBlock   = arrayfun(@(x)1:x, blockSize, 'UniformOutput', false);
    firstBlock   = A(firstBlock{:});
    
    outputBlockSize = size(firstBlock);
    
    
    
    if ~isempty(fcn)
        try
            if isempty(B)
                firstBlock = fcn(firstBlock);
            else                
                firstBlock = bsxfun(fcn, firstBlock, B);
            end
            outputBlockSize = size(firstBlock);
            
        catch ME
        end
    end
    
    % The size of the blocks will be preserved if requesting a 'split'
    % (cell array of (processed) blocks) or if the results will be
    % concatenated in a dimension greater than the dimensionality of A.
    preserveSize = dim > ndims(A);
    
    % Initialize output
    cellCapture = split;
    if cellCapture
        outArg = cell(blocksPerDim);
    elseif preserveSize
        %outArg = zeros(outputBlockSize .* blocksPerDim);
        
        
    end
    
    
    blockSize
    dim       
    fcn       
    B         
    split     
    
    
    
    outArg
    
    
        
    
    
    %% 
    
end


